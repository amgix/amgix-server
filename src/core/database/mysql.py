"""
MySQL database implementation.
Subclasses MariaDatabase since MySQL and MariaDB share most functionality.
"""

from .mariadb import MariaDatabase


class MySQLDatabase(MariaDatabase):
    """
    MySQL database implementation.
    
    Currently inherits all functionality from MariaDatabase since MySQL doesn't support
    dense vectors yet. When MySQL adds dense vector support, this class can override
    specific methods to handle MySQL-specific syntax and features.
    """
    
    def __init__(self, connection_string: str, logger, **kwargs):
        super().__init__(connection_string, logger=logger, **kwargs)
        
        # Override SQL templates with MySQL-specific syntax (merge with base templates)
        mysql_templates = {
        }
        
        # Merge MySQL-specific templates with base templates
        self.SQL_TEMPLATES.update(mysql_templates)
