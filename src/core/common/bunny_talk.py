from datetime import datetime, date, timezone
import json
import asyncio
import contextvars
import inspect
from logging import Logger
import os
import time
import traceback
import uuid
import random
from typing import Any, Dict, Tuple, Union, Optional, get_origin, get_args, get_type_hints
from types import UnionType as _UnionType

from pydantic import BaseModel, TypeAdapter
from aio_pika import DeliveryMode, ExchangeType, Message, RobustConnection, RobustChannel, RobustExchange, RobustQueue, connect_robust
from aio_pika.exceptions import AMQPException, ChannelNotFoundEntity
from src.core.common.constants import APP_PREFIX, MAX_QUEUE_MESSAGES, MAX_QUEUE_SIZE_BYTES, RPC_TIMEOUT_SECONDS, MAX_DB_RETRIES

# Get hostname for tracing
HOSTNAME = os.getenv('HOSTNAME', 'unknown')

# Context variables for automatic trace propagation (namespaced to avoid conflicts)
trace_id_var = contextvars.ContextVar(f'{APP_PREFIX}_trace_id', default=None)
trace_chain_var = contextvars.ContextVar(f'{APP_PREFIX}_trace_chain', default=[])
trace_meta_var = contextvars.ContextVar(f'{APP_PREFIX}_trace_meta', default={})


class BunnyJsonEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, datetime):
            return o.isoformat()
        elif isinstance(o, BaseModel):
            return o.model_dump()
        return super().default(o)


class RPCResponse(BaseModel):
    """Internal response class for RPC calls that can contain either a result or an error."""
    success: bool
    result: Optional[Any] = None
    error: Optional[str] = None
    error_type: Optional[str] = None  # Full exception class path for re-raising on caller side
    result_type: Optional[str] = None  # Full class path for deserialization
    
    @classmethod
    def success_response(cls, result: Any) -> 'RPCResponse':
        """Create a successful RPC response."""
        result_type = None
        if isinstance(result, BaseModel):
            # Store the full class path for deserialization
            result_type = f"{result.__class__.__module__}.{result.__class__.__name__}"
        return cls(success=True, result=result, result_type=result_type)
    
    @classmethod
    def error_response(cls, error: Union[str, Exception]) -> 'RPCResponse':
        """Create an error RPC response."""
        if isinstance(error, Exception):
            error_type = f"{error.__class__.__module__}.{error.__class__.__name__}"
            return cls(success=False, error=str(error), error_type=error_type)
        return cls(success=False, error=str(error))
    


QUORUM_QUEUE_ARGUMENTS = {
        'x-queue-type': 'quorum',  # Make it a quorum queue for HA
        'x-delivery-limit': MAX_DB_RETRIES + 1,  # Max retries for DB errors (1 initial + MAX_DB_RETRIES redeliveries)
        'x-max-length': MAX_QUEUE_MESSAGES, # Max 10000 messages in queue
        'x-max-length-bytes': MAX_QUEUE_SIZE_BYTES, # Max 1GB in queue
        'x-overflow': 'reject-publish',
    }

CLASSIC_QUEUE_ARGUMENTS = {
        'x-queue-type': 'classic',  
        # 'x-delivery-limit': MAX_QUEUE_DELIVERY_ATTEMPTS,  # Max 4 delivery attempts (1 initial + 3 retries)
        'x-max-length': MAX_QUEUE_MESSAGES, # Max 10000 messages in queue
        'x-max-length-bytes': MAX_QUEUE_SIZE_BYTES, # Max 1GB in queue
        'x-overflow': 'reject-publish',
        'x-expires': max((RPC_TIMEOUT_SECONDS + 5) * 1000, 10000)
    }

# Types that need coercion when crossing the JSON boundary (plain primitives are pass-through)
_COERCE_TYPES = (datetime, date, BaseModel)

def _needs_coercion(annotation: Any) -> bool:
    """Return True if the annotation requires TypeAdapter coercion from JSON."""
    if annotation is None or annotation is type(None):
        return False
    origin = get_origin(annotation)
    # Unwrap Optional[X] (typing.Union) and X | Y (types.UnionType, Python 3.10+)
    if origin is Union or isinstance(annotation, _UnionType):
        args = [a for a in get_args(annotation) if a is not type(None)]
        return any(_needs_coercion(a) for a in args)
    if origin is list:
        args = get_args(annotation)
        return bool(args) and _needs_coercion(args[0])
    if inspect.isclass(annotation):
        return issubclass(annotation, _COERCE_TYPES)
    return False

def _make_adapter_if_needed(annotation: Any) -> Optional[TypeAdapter]:
    """Return a TypeAdapter for types needing coercion, or None for fast pass-through."""
    if _needs_coercion(annotation):
        return TypeAdapter(annotation)
    return None


class BunnyTalk:
    """
    BunnyTalk is a class that provides a simple interface for publishing and consuming messages from RabbitMQ.
    """
    
    def __init__(self, logger: Logger, connection: RobustConnection,
                 publish_channel: RobustChannel, publish_exchange: RobustExchange,
                 passive_check_channel: RobustChannel):
        self.logger = logger
        self.connection = connection
        self.publish_channel = publish_channel
        self.publish_exchange = publish_exchange
        self.passive_check_channel = passive_check_channel
        self.consumers = []  # Track consumers for cleanup
        self._rpc_futures: Dict[str, asyncio.Future] = {}
        self._rpc_reply_ctx: Dict[str, Tuple[Optional[str], list, dict]] = {}
        self._type_cache: Dict[str, type] = {}  # Cache for dynamically imported types
        self.reply_queue_name = None
        self.reply_queue = None
        self.reply_consumer_tag = None
        self.lock_channel = None
        self._queue_exchanges: Dict[str, RobustExchange] = {}  # Track exchanges per queue_name
        self._queue_channels: Dict[str, RobustChannel] = {}  # Track channels per queue_name
        self._active_consumers: Dict[Tuple[str, str], Tuple[RobustQueue, str]] = {}  # Track consumers by (queue_name, handler_name)
    
    @staticmethod
    def get_trace_context() -> Dict[str, Any]:
        """Get the current trace context as a dictionary."""
        return {
            'trace_id': trace_id_var.get(),
            'trace_chain': trace_chain_var.get(),
            'trace_meta': trace_meta_var.get()
        }

    def log_trace_context(self, prefix: str):
        """Log the current trace context as a dictionary."""
        self.logger.debug(
            f"{prefix} trace_id: {trace_id_var.get()}, trace_chain: {trace_chain_var.get()}, trace_meta: {trace_meta_var.get()}"
        )

    def get_broker_version(self) -> Optional[str]:
        """Return the AMQP broker version from Connection.Start ``server_properties`` (e.g. RabbitMQ).

        Uses the underlying :class:`aiormq.connection.Connection` on the robust transport
        (``robust_connection.transport.connection.server_properties``), i.e. the broker
        ``version`` field from the Connection.Start handshake (as in blocking clients'
        ``connection.server_properties['version']``).
        """
        conn = self.connection
        if conn.is_closed:
            return None
        transport = conn.transport
        if transport is None:
            return None
        under = transport.connection
        props = getattr(under, "server_properties", None)
        if not props:
            return None
        raw = props.get("version")
        if raw is None:
            raw = props.get(b"version")
        if raw is None:
            return None
        if isinstance(raw, (bytes, bytearray, memoryview)):
            return bytes(raw).decode("utf-8", errors="replace")
        return str(raw)

    @classmethod
    async def create(cls, logger: Logger, amqp_url: str, max_retries: int = 10, retry_delay: float = 5.0):
        """Create BunnyTalk with retry logic for RabbitMQ startup."""
        for attempt in range(max_retries):
            try:
                connection = await connect_robust(amqp_url, heartbeat=RPC_TIMEOUT_SECONDS / 2 + 5)

                # Channel for publishing and RPC reply consumers. Enable on_return_raises.
                publish_channel = await connection.channel(on_return_raises=True)
                publish_exchange = await publish_channel.declare_exchange(
                    f'{APP_PREFIX}.topic', ExchangeType.TOPIC, durable=True
                )

                # Dedicated channel for passive queue checks (to avoid breaking publish_channel)
                passive_check_channel = await connection.channel()

                # Instantiate BunnyTalk
                instance = cls(logger, connection, publish_channel, publish_exchange, passive_check_channel)

                # Set up a single, long-lived exclusive reply queue on the publish channel
                instance._rpc_futures: Dict[str, asyncio.Future] = {}
                instance.reply_queue_name = f"{APP_PREFIX}-rpc-{HOSTNAME}-{str(uuid.uuid4())[:8]}"
                # Declare and consume reply queue directly on publish channel (raw message handler)
                instance.reply_queue = await publish_channel.declare_queue(
                    instance.reply_queue_name,
                    durable=True,
                    exclusive=True,
                    auto_delete=True,
                    arguments=CLASSIC_QUEUE_ARGUMENTS
                )
                await instance.reply_queue.bind(publish_exchange, routing_key=instance.reply_queue_name)
                instance.reply_consumer_tag = await instance.reply_queue.consume(instance._on_rpc_reply)

                return instance
            except Exception as e:
                if attempt < max_retries - 1:
                    logger.error(f"Failed to connect to RabbitMQ (attempt {attempt + 1}/{max_retries}): {e}")
                    logger.error(f"Retrying in {retry_delay} seconds...")
                    await asyncio.sleep(retry_delay)
                else:
                    raise Exception(f"Failed to connect to RabbitMQ after {max_retries} attempts: {e}")
        
    async def listen(self, routing_key: str, handler: callable, queue_name: str = None,
                            exclusive: bool = False,
                            passive: bool = False,
                            auto_delete: bool = False,
                            prefetch_count: int = 1,
                            robust: bool = True,
                            consumer_priority: int = None,
                            single_active_consumer: bool = False):

        if queue_name is None:
            queue_name = f'{APP_PREFIX}-{routing_key}'

        queue_arguments = QUORUM_QUEUE_ARGUMENTS if not auto_delete and not exclusive else CLASSIC_QUEUE_ARGUMENTS
        
        if single_active_consumer:
            queue_arguments = {**queue_arguments, "x-single-active-consumer": True}

        # Create dedicated channel per queue for true parallelism
        existing = self._queue_channels.get(queue_name)
        if existing is None or existing.is_closed:
            channel = await existing.reopen() if existing else await self.connection.channel()
            # Set QoS with configurable prefetch for this queue
            await channel.set_qos(prefetch_count=prefetch_count)
            self._queue_channels[queue_name] = channel

            if not existing:
                exchange = await channel.declare_exchange(
                    f'{APP_PREFIX}.topic', ExchangeType.TOPIC, durable=True
                )
                self._queue_exchanges[queue_name] = exchange
            else:
                exchange = self._queue_exchanges[queue_name]
        else:
            channel = existing
            exchange = self._queue_exchanges[queue_name]


        # Declare queue with handling for exclusive declarations that may close the channel
        try:
            queue: RobustQueue = await channel.declare_queue(
                queue_name,
                durable=True,
                exclusive=exclusive,
                passive=passive,
                auto_delete=auto_delete,
                arguments=queue_arguments,
                robust=robust,
            )
        except Exception as e:
            if exclusive and "resource_locked" in str(e).casefold():
                self.logger.debug(f"Error declaring queue {queue_name}: {e}")
            else:
                self.logger.error(f"Error declaring queue {queue_name}: {e}")
            # # On any declare error, if this was an exclusive listener attempt,
            # # ensure we don't retain/keep using a bad/closed channel.
            # if exclusive:
            #     try:
            #         if channel and not channel.is_closed:
            #             await channel.close()
            #     except Exception:
            #         pass
            #     try:
            #         if self._exclusive_channels.get(queue_name) is channel:
            #             del self._exclusive_channels[queue_name]
            #     except Exception:
            #         pass
            # Re-raise so caller can decide next steps
            raise e

        await queue.bind(exchange, routing_key=routing_key)

        # Get handler signature information once
        sig = inspect.signature(handler)
        is_async = inspect.iscoroutinefunction(handler)
        
        # Resolve parameter names and types once (performance)
        param_names = list(sig.parameters.keys())
        
        # Resolve all type hints once using typing.get_type_hints
        hints = get_type_hints(handler, globalns=getattr(handler, '__globals__', None))
        # Map param name → TypeAdapter for types that need coercion, None for pass-through primitives
        param_adapters: Dict[str, Optional[TypeAdapter]] = {}
        for name, param in sig.parameters.items():
            annotation = hints.get(name, inspect.Parameter.empty)
            if annotation == inspect.Parameter.empty:
                raise TypeError(f"Missing or unresolvable type annotation for parameter '{name}' in handler '{handler.__name__}'")
            param_adapters[name] = _make_adapter_if_needed(annotation)

        async def process_message(message: Message):
            try:
                payload = json.loads(message.body.decode())
                
                # Always extract trace info to maintain chain continuity
                trace_id = message.headers.get(f'{APP_PREFIX}_trace_id') if message.headers else None
                trace_chain = json.loads(message.headers.get(f'{APP_PREFIX}_trace_chain', '[]')) if message.headers else []
                trace_meta = json.loads(message.headers.get(f'{APP_PREFIX}_trace_meta', '{}')) if message.headers else {}
                
                # Add structured call information to chain
                call_info = {
                    'hostname': HOSTNAME,
                    'function': handler.__name__,
                    'timestamp': datetime.now(timezone.utc).isoformat(),
                    'call_id': str(uuid.uuid4())
                }
                trace_chain = trace_chain + [call_info]
                
                # Set context variables for automatic propagation
                trace_id_var.set(trace_id)
                trace_chain_var.set(trace_chain)
                trace_meta_var.set(trace_meta)
                
                if message.reply_to:
                    # This is an RPC call, send response back
                    try:
                        # Automatically map parameters to handler function
                        result = await self._call_handler_with_params_local(handler, payload, param_names, param_adapters, is_async)
                        # Wrap result in RPCResponse
                        if isinstance(result, RPCResponse):
                            response = result
                        else:
                            response = RPCResponse.success_response(result)
                    except Exception as e:
                        self.logger.error(f"Error calling handler for queue message: {e}")
                        self.logger.debug(f"Full traceback: {traceback.format_exc()}")
                        # Wrap error in RPCResponse
                        response = RPCResponse.error_response(e)
                    
                    # Publish RPC response directly with original correlation_id
                    response_payload = {
                        'args': [],
                        'kwargs': {'response': response}
                    }
                    await self.publish_exchange.publish(
                        Message(
                            json.dumps(response_payload, cls=BunnyJsonEncoder).encode(),
                            correlation_id=message.correlation_id,
                            delivery_mode=DeliveryMode.NOT_PERSISTENT,
                            headers={
                                f'{APP_PREFIX}_trace_id': trace_id_var.get(),
                                f'{APP_PREFIX}_trace_chain': json.dumps(trace_chain_var.get() or []),
                                f'{APP_PREFIX}_trace_meta': json.dumps(trace_meta_var.get() or {})
                            }
                        ),
                        routing_key=message.reply_to,
                        timeout=5
                    )
                else:
                    # Regular event, automatically map parameters to handler function
                    await self._call_handler_with_params_local(handler, payload, param_names, param_adapters, is_async)
                
                await message.ack()
            except Exception as e:
                self.logger.error(f"Error processing queue message: {e}")
                if message.reply_to:
                    await message.reject(requeue=False)
                else:
                    await message.reject(requeue=True)
                    
        # Check if we already have a consumer for this queue+handler combination
        consumer_key = (queue_name, handler.__name__)
        if consumer_key in self._active_consumers:
            return self._active_consumers[consumer_key]

        # Retry consume for quorum queues (RabbitMQ 4.x Ra initialization race)
        max_consume_attempts = 20
        for attempt in range(max_consume_attempts):
            try:
                consume_kwargs = {}
                if consumer_priority is not None:
                    consume_kwargs["arguments"] = {"x-priority": consumer_priority}
                consumer_tag = await queue.consume(process_message, **consume_kwargs)
                break
            except Exception as e:
                error_text = str(e).casefold()
                should_retry = ("noproc" in error_text)
                if should_retry and attempt < max_consume_attempts - 1:
                    self.logger.warning(f"Queue {queue_name} not ready (attempt {attempt + 1}/{max_consume_attempts}), retrying...")

                    # brief backoff
                    await asyncio.sleep(0.2)

                    # If the broker closed the connection, wait for robust reconnect
                    if self.connection.is_closed:
                        for _ in range(10):  # up to ~2s
                            if not self.connection.is_closed:
                                break
                            await asyncio.sleep(0.2)

                    # Reopen the same channel if needed
                    if channel.is_closed:
                        await channel.reopen()
                    continue
                else:
                    raise
        
        self.consumers.append((queue, consumer_tag))  # Track for cleanup
        self._active_consumers[consumer_key] = (queue, consumer_tag)  # Track by queue+handler

        return queue, consumer_tag

    async def get_queue_consumers(self, queue_name: str) -> int:
        """
        Check how many consumers a queue has using passive declaration.
        
        Args:
            queue_name: Name of the queue to check
            
        Returns:
            Number of consumers, or 0 if queue doesn't exist
        """
        try:
            # Use dedicated passive_check_channel (passive=True closes channel on error)
            # Use the underlying aiormq channel to avoid RobustChannel trying to "restore"
            # a missing queue in its recovery loop, which causes a crash loop.
            queue = await self.passive_check_channel.channel.queue_declare(
                f"{APP_PREFIX}-{queue_name}",
                passive=True,
                timeout=2
            )
            # Get queue declaration result which contains consumer_count
            return queue.consumer_count
        except ChannelNotFoundEntity:
            # Queue doesn't exist (404 NOT_FOUND)
            return 0
        except Exception as e:
            # aio_pika/aiormq sometimes raises "Channel closed by RPC timeout" or other state errors
            # instead of ChannelNotFoundEntity due to race conditions when the broker closes the channel on 404.
            error_str = str(e).casefold()
            if "channel closed" in error_str or "not_found" in error_str or "rpc timeout" in error_str:
                self.logger.debug(f"Queue check failed for {queue_name} (likely missing): {e}")
                return 0
            raise

    async def _on_rpc_reply(self, message: Message):
        future: asyncio.Future = None
        try:
            correlation_id = message.correlation_id
            future = self._rpc_futures.pop(correlation_id, None)
            if future is None:
                self.logger.debug(
                    f"RPC reply received with unknown correlation_id={correlation_id}; "
                    f"queue={self.reply_queue_name}. Dropping."
                )
                await message.ack()
                return

            # Extract trace info from reply headers and update context variables
            if message.headers:
                trace_id = message.headers.get(f'{APP_PREFIX}_trace_id')
                trace_chain = json.loads(message.headers.get(f'{APP_PREFIX}_trace_chain', '[]'))
                trace_meta = json.loads(message.headers.get(f'{APP_PREFIX}_trace_meta', '{}'))
                # Stash for the awaiting rpc() caller task to adopt
                self._rpc_reply_ctx[correlation_id] = (trace_id, trace_chain, trace_meta)
                # Also set in this consumer task (harmless, but primary adoption happens in rpc())
                trace_id_var.set(trace_id)
                trace_chain_var.set(trace_chain)
                trace_meta_var.set(trace_meta)

            payload = json.loads(message.body.decode())
            response_dict = payload.get('kwargs', {}).get('response')
            response = RPCResponse.model_validate(response_dict)
            future.set_result(response)
            await message.ack()
        except Exception as e:
            self.log_trace_context(f"Error processing RPC reply: {e}")
            self.logger.debug(f"Full traceback: {traceback.format_exc()}")
            if future is not None and not future.done():
                future.set_exception(e)
            try:
                await message.ack()
            except Exception as ack_err:
                self.logger.warning(f"Failed to ack RPC reply message after error: {ack_err}")
    
    async def _call_handler_with_params_local(
        self,
        handler: callable,
        payload: Dict[str, Any],
        param_names: list,
        param_adapters: Dict[str, TypeAdapter],
        is_async: bool,
    ) -> Any:
        """
        Map JSON payload fields to the handler, using Pydantic adapters so primitives like
        `datetime` round-trip through JSON (ISO strings) back into typed Python values.
        """
        args = payload.get('args', [])
        kwargs = payload.get('kwargs', {})

        def convert_value(param_name: str, value: Any) -> Any:
            adapter = param_adapters.get(param_name)
            if not adapter:
                return value
            return adapter.validate_python(value)

        # Map positional args
        bound_args: Dict[str, Any] = {}
        for i, arg in enumerate(args):
            if i < len(param_names):
                param_name = param_names[i]
                bound_args[param_name] = convert_value(param_name, arg)

        # Map keyword args
        for key, value in kwargs.items():
            bound_args[key] = convert_value(key, value)

        if is_async:
            return await handler(**bound_args)
        else:
            return handler(**bound_args)
        
    async def talk(self, routing_key: str, persistent: bool = True, *args, trace_meta: Optional[Dict[str, Any]] = None, start_trace: bool = False, **kwargs):
        """
        Publish event with automatically serialized parameters.
        
        Args:
            routing_key: Routing key for the event
            *args: Positional arguments to serialize
            **kwargs: Keyword arguments to serialize
        """
        # Get trace info from context or create new
        if start_trace:
            # Start fresh trace chain
            trace_id = str(uuid.uuid4())
            trace_chain = []
        else:
            # Continue existing trace chain
            trace_id = trace_id_var.get() or str(uuid.uuid4())
            trace_chain = trace_chain_var.get() or []
        
        # Add structured call information to chain
        calling_func = inspect.currentframe().f_back.f_code.co_name
        call_info = {
            'hostname': HOSTNAME,
            'function': f"{calling_func}->talk->{routing_key}",
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'call_id': str(uuid.uuid4())
        }
        trace_chain = trace_chain + [call_info]
        
        # Set context variables for automatic propagation
        trace_id_var.set(trace_id)
        trace_chain_var.set(trace_chain)
        trace_meta_var.set((trace_meta_var.get() or {}) if trace_meta is None else {**(trace_meta_var.get() or {}), **trace_meta})
        
        # Create payload from args and kwargs
        payload = {
            'args': args,
            'kwargs': kwargs
        }
        
        await self.publish_exchange.publish(
            Message(
                json.dumps(payload, cls=BunnyJsonEncoder).encode(), 
                delivery_mode=DeliveryMode.PERSISTENT if persistent else DeliveryMode.NOT_PERSISTENT,
                headers={
                    f'{APP_PREFIX}_trace_id': trace_id,
                    f'{APP_PREFIX}_trace_chain': json.dumps(trace_chain),
                    f'{APP_PREFIX}_trace_meta': json.dumps((trace_meta_var.get() or {}) if trace_meta is None else {**(trace_meta_var.get() or {}), **trace_meta})
                }
            ),
            routing_key=routing_key,
            timeout=5
        )
        
    async def rpc(self, routing_key: str, timeout: float = RPC_TIMEOUT_SECONDS, *args, trace_meta: Optional[Dict[str, Any]] = None, start_trace: bool = False, return_type: Optional[type] = None, **kwargs) -> Any:
        """
        Publish RPC request and wait for response.
        
        Args:
            routing_key: Routing key for the RPC call
            timeout: Timeout in seconds
            *args: Positional arguments to pass to the RPC handler
            **kwargs: Keyword arguments to pass to the RPC handler
            
        Returns:
            Response data on success
            
        Raises:
            Exception: If the RPC call failed
            
        Raises:
            asyncio.TimeoutError: If no response received within timeout
        """
        correlation_id = str(uuid.uuid4())
        reply_to = self.reply_queue_name
        
        # Get trace info from context or create new
        if start_trace:
            # Start fresh trace chain
            trace_id = str(uuid.uuid4())
            trace_chain = []
        else:
            # Continue existing trace chain
            trace_id = trace_id_var.get() or str(uuid.uuid4())
            trace_chain = trace_chain_var.get() or []
        
        # Add structured call information to chain
        calling_func = inspect.currentframe().f_back.f_code.co_name
        call_info = {
            'hostname': HOSTNAME,
            'function': f"{calling_func}->rpc->{routing_key}",
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'call_id': str(uuid.uuid4())
        }
        trace_chain = trace_chain + [call_info]
        
        # Set context variables for automatic propagation
        trace_id_var.set(trace_id)
        trace_chain_var.set(trace_chain)
        trace_meta_var.set((trace_meta_var.get() or {}) if trace_meta is None else {**(trace_meta_var.get() or {}), **trace_meta})
        
        # Create payload from args and kwargs
        payload = {
            'args': args,
            'kwargs': kwargs
        }
        
        # Set up response handler
        response_future = asyncio.Future()
        
        # Register future in the dispatcher before publish
        self._rpc_futures[correlation_id] = response_future
        
        try:
            # Publish the request
            await self.publish_exchange.publish(
                Message(
                    json.dumps(payload, cls=BunnyJsonEncoder).encode(),
                    reply_to=reply_to,
                    correlation_id=correlation_id,
                    expiration=timeout + 5,
                    headers={
                        f'{APP_PREFIX}_trace_id': trace_id,
                        f'{APP_PREFIX}_trace_chain': json.dumps(trace_chain),
                        f'{APP_PREFIX}_trace_meta': json.dumps((trace_meta_var.get() or {}) if trace_meta is None else {**(trace_meta_var.get() or {}), **trace_meta})
                    }
                ),
                routing_key=routing_key,
                timeout=5
            )
            
            # Wait for response
            try:
                response = await asyncio.wait_for(response_future, timeout=timeout)
            except asyncio.TimeoutError:
                raise TimeoutError(f"RPC call to '{routing_key}' timed out after {timeout} seconds")
            
            # Adopt reply trace context into the caller task context
            reply_ctx = self._rpc_reply_ctx.pop(correlation_id, None)
            if reply_ctx is not None:
                reply_trace_id, reply_trace_chain, reply_trace_meta = reply_ctx
                trace_id_var.set(reply_trace_id)
                trace_chain_var.set(reply_trace_chain)
                trace_meta_var.set(reply_trace_meta)

            # Check if it's an error response and raise exception
            if not response.success:
                if response.error_type:
                    error_class = None
                    try:
                        module_path, class_name = response.error_type.rsplit(".", 1)
                        module = __import__(module_path, fromlist=[class_name])
                        error_class = getattr(module, class_name)
                    except Exception:
                        # Fall back to generic exception if dynamic reconstruction fails
                        error_class = None
                    if inspect.isclass(error_class) and issubclass(error_class, Exception):
                        raise error_class(response.error)
                raise Exception(response.error)
            
            # Convert result back to expected type if needed
            result = response.result
            
            # Deserialize based on result_type metadata
            if response.result_type and isinstance(result, dict):
                # Check cache first
                result_class = self._type_cache.get(response.result_type)
                
                if result_class is None:
                    try:
                        # Import the class dynamically and cache it
                        module_path, class_name = response.result_type.rsplit('.', 1)
                        module = __import__(module_path, fromlist=[class_name])
                        result_class = getattr(module, class_name)
                        
                        if inspect.isclass(result_class) and issubclass(result_class, BaseModel):
                            self._type_cache[response.result_type] = result_class
                        else:
                            result_class = None
                    except Exception as e:
                        self.logger.warning(f"RPC {routing_key}: Failed to import result_type={response.result_type}: {e}")
                
                if result_class:
                    self.logger.debug(f"RPC {routing_key}: Converting dict to {result_class}")
                    result = result_class.model_validate(result)
            
            # Fall back to explicitly provided return_type if dynamic import failed
            if isinstance(result, dict) and return_type:
                if inspect.isclass(return_type) and issubclass(return_type, BaseModel):
                    self.logger.debug(f"RPC {routing_key}: Converting dict to {return_type} based on explicit return_type")
                    result = return_type.model_validate(result)
            
            return result

        except Exception as e:
            # Catch-all to log trace for any unexpected error path
            self.log_trace_context(
                f"RPC exception: routing_key={routing_key}, correlation_id={correlation_id}, reply_to={reply_to}, error={e}"
            )
            raise

        finally:
            # Cleanup the future if it wasn't consumed
            self._rpc_futures.pop(correlation_id, None)
            # Ensure no leaked reply context for this correlation id
            self._rpc_reply_ctx.pop(correlation_id, None)

        
    async def cancel_listener(self, queue: RobustQueue, consumer_tag: str) -> None:
        """Cancel and unregister a consumer previously created by listen()."""
        try:
            await queue.cancel(consumer_tag)
        except Exception as e:
            self.logger.error(f"Error canceling consumer {consumer_tag} for queue {getattr(queue, 'name', '<unknown>')}: {e}")
            raise

        # Remove from global consumers list
        self.consumers = [(q, ct) for (q, ct) in self.consumers if not (q is queue and ct == consumer_tag)]

        # Remove any matching entries from active_consumers
        try:
            keys_to_delete = [key for key, val in self._active_consumers.items() if val == (queue, consumer_tag)]
            for key in keys_to_delete:
                del self._active_consumers[key]
        except Exception as e:
            # Log but don't fail cancel flow since the consumer is already canceled
            self.logger.error(f"Error unregistering consumer {consumer_tag} for queue {getattr(queue, 'name', '<unknown>')}: {e}")
            raise

    
    # def lock(self, lock_name: str, acquire_timeout: float = 5.0, retry_backoff: float = 0.1):
    #     """
    #     Acquire a distributed lock using an exclusive queue.
        
    #     Args:
    #         lock_name: Name of the lock
    #         acquire_timeout: How long to keep trying to obtain the lock (seconds)
    #         retry_backoff: Seconds to sleep between retries
            
    #     Returns:
    #         Async context manager for the lock
    #     """
    #     return BunnyLockContext(self.logger, self, lock_name, acquire_timeout, retry_backoff)
    
    async def close(self):
        """Close the connection and all consumers gracefully."""
        # Cancel all consumers
        for queue, consumer_tag in self.consumers:
            try:
                await queue.cancel(consumer_tag)
            except Exception as e:
                self.logger.error(f"Error canceling consumer {consumer_tag} for queue {queue.name}: {e}")
                pass

        self.consumers.clear()
        self._active_consumers.clear()

        # Cancel single reply consumer
        try:
            if self.reply_queue and self.reply_consumer_tag:
                await self.reply_queue.cancel(self.reply_consumer_tag)
        except Exception as e:
            self.logger.error(f"Error canceling reply consumer {self.reply_consumer_tag} for queue {self.reply_queue.name}: {e}")
            pass
        
        # Close publish channel
        try:
            if not self.publish_channel.is_closed:
                await self.publish_channel.close()
        except Exception as e:
            self.logger.error(f"Error closing publish channel: {e}")
            pass
        
        # Close passive check channel
        try:
            if not self.passive_check_channel.is_closed:
                await self.passive_check_channel.close()
        except Exception as e:
            self.logger.error(f"Error closing passive check channel: {e}")
            pass
        
        # Close per-queue channels
        for qname, ch in list(self._queue_channels.items()):
            try:
                if ch and not ch.is_closed:
                    await ch.close()
            except Exception as e:
                self.logger.error(f"Error closing queue channel for {qname}: {e}")
        self._queue_channels.clear()
        self._queue_exchanges.clear()
        # Close lock channel
        try:
            if self.lock_channel and not self.lock_channel.is_closed:
                await self.lock_channel.close()
        except Exception as e:
            self.logger.error(f"Error closing lock channel: {e}")
            pass

        # Close connection
        if self.connection:
            try:
                await self.connection.close()
            except Exception as e:
                self.logger.error(f"Error closing connection: {e}")
                pass


# class BunnyLockContext:
#     """Async context manager for distributed locking using exclusive queues."""

#     MAX_CHANNEL_POOL_SIZE = 255

#     _channel_pool: Dict[int, Tuple[RobustChannel, bool, float]] = {}
#     _channel_pool_lock = asyncio.Lock()
    
#     def __init__(self, logger: Logger, bunny_talk: 'BunnyTalk', lock_name: str, acquire_timeout: float, retry_backoff: float):
#         self.logger = logger
#         self.bunny_talk = bunny_talk
#         self.lock_name = lock_name
#         self.acquire_timeout = acquire_timeout
#         self.retry_backoff = retry_backoff
#         self.queue: Optional[RobustQueue] = None  # Holds the exclusive queue while lock is owned
#         self.lock_channel: Optional[RobustChannel] = None  # Channel used for lock acquisition
        
#     async def __aenter__(self):
#         """Acquire the lock by declaring an exclusive queue using add_consumer."""

#         self.logger.debug(f"BunnyLockContext: Acquiring lock '{self.lock_name}'")

#         deadline = time.time() + self.acquire_timeout

#         while True:
#             async with self._channel_pool_lock:

#                 # t=time.time()
#                 # size = len(self._channel_pool)
#                 # old = [int(t-c[2]) for c in self._channel_pool.values() if c[1] and c[2] and t-c[2] > 5]
#                 # if size > 101 or old:
#                 #     self.logger.info(f"BunnyLockContext: __aenter__ pool size {size}, old: {old}")
#                 #     # exit(1)

#                 self.lock_channel = next((c[0] for c in self._channel_pool.values() if not c[1]), None)

#                 if self.lock_channel is None and len(self._channel_pool) < self.MAX_CHANNEL_POOL_SIZE:
#                     try:
#                         self.lock_channel = await self.bunny_talk.connection.channel()
#                         # self._channel_pool[id(self.lock_channel)] = (self.lock_channel, True)
#                     except Exception as e:
#                         self.logger.error(f"BunnyLockContext: Error acquiring communication channel for lock '{self.lock_name}': {e}")

#                 if self.lock_channel is not None:
#                     self.logger.debug(f"BunnyLockContext: Successfully acquired communication channel {id(self.lock_channel)} for lock '{self.lock_name}'")
#                     self._channel_pool[id(self.lock_channel)] = (self.lock_channel, True, time.time())
#                     break

#             if time.time() > deadline:
#                 self.logger.debug(f"BunnyLockContext: Could not acquire communication channel for lock '{self.lock_name}' within {self.acquire_timeout:.1f}s")
#                 raise TimeoutError(
#                     f"Could not acquire communication channel for lock '{self.lock_name}' within {self.acquire_timeout:.1f}s"
#                 )

#             await asyncio.sleep(self.retry_backoff)

#         count = 0
#         while time.time() < deadline:
#             try:
#                 queue_name = f"{APP_PREFIX}-lock-{self.lock_name}"

#                 # # Use a fresh, short-lived channel for lock acquisition to avoid interference
#                 # if self.lock_channel.is_closed:
#                 #     await self.lock_channel.reopen()

#                 # Declare an exclusive, auto-delete queue on this channel.
#                 # This will fail with a channel error if another connection owns it.
#                 self.queue = await self.lock_channel.declare_queue(
#                     queue_name,
#                     durable=False,
#                     exclusive=True,
#                     auto_delete=True,
#                     timeout=5.0,
#                     arguments=CLASSIC_QUEUE_ARGUMENTS
#                 )

#                 # Success - we own the lock
#                 self.logger.debug(f"BunnyLockContext: Successfully acquired lock '{self.lock_name}'")
#                 return self

#             except Exception as e:
#                 error_msg = str(e)
                
#                 if not "resource_locked" in error_msg.casefold():
#                     self.logger.debug(f"BunnyLockContext: Error trying to acquire lock '{self.lock_name}': {e}")

#                 # backoff and retry
#                 count += 1
#                 backoff = min(count * self.retry_backoff, 0.8)
#                 await asyncio.sleep(backoff)
#                 continue
        
#         async with self._channel_pool_lock:
#             self._channel_pool[id(self.lock_channel)] = (self.lock_channel, False, 0)
#         self.logger.debug(f"BunnyLockContext: Could not acquire lock '{self.lock_name}' within {self.acquire_timeout:.1f}s")
#         raise TimeoutError(
#             f"Could not acquire lock '{self.lock_name}' within {self.acquire_timeout:.1f}s"
#         )
    
#     async def __aexit__(self, exc_type, exc_val, exc_tb):
#         """Release the lock by canceling the consumer."""

#         self.logger.debug(f"BunnyLockContext: Releasing lock '{self.lock_name}'")

#         if self.queue:
#             try:
#                 self.logger.debug(f"BunnyLockContext: Deleting lock queue '{self.lock_name}'")
#                 await self.lock_channel.queue_delete(self.queue.name, if_unused=False, if_empty=False, timeout=5.0)
#                 self.logger.debug(f"BunnyLockContext: Deleted lock queue '{self.lock_name}'")
#             except Exception as e:
#                 # Ignore errors releasing the lock
#                 self.logger.debug(f"BunnyLockContext: Error releasing lock '{self.lock_name}': {e}")
#             finally:
#                 self.queue = None
        
#         # Close the lock channel to free resources
#         if self.lock_channel:
#             # try:
#             #     self.logger.debug(f"BunnyLockContext: Closing lock channel '{self.lock_name}'")
#             #     if not self.lock_channel.is_closed:
#             #         await self.lock_channel.close()


#             # except Exception as e:
#             #     self.logger.debug(f"BunnyLockContext: Error closing lock channel for '{self.lock_name}': {e}")

#             # finally:
#             async with self._channel_pool_lock:
#                 # self.logger.info(f"BunnyLockContext: Releasing lock channel for '{self.lock_name}'")
#                 self._channel_pool[id(self.lock_channel)] = (self.lock_channel, False, 0)

#             self.lock_channel = None
        
#         return False
