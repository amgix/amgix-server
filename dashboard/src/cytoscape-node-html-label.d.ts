declare module 'cytoscape-node-html-label' {
  import type cytoscape from 'cytoscape'

  export default function register(cy: typeof cytoscape): void
}
