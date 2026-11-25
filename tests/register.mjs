// The app's sources use Vite-style extensionless imports ('../voxel/ChunkData').
// Node's ESM resolver needs the extension, so a relative import that is not found
// is retried with '.js'. Loaded with `node --import ./tests/register.mjs`.
import * as nodeModule from 'node:module';

const withJs = (specifier, err) => {
  if (err?.code !== 'ERR_MODULE_NOT_FOUND' || !specifier.startsWith('.') || specifier.endsWith('.js')) throw err;
  return `${specifier}.js`;
};

if (nodeModule.registerHooks) {
  nodeModule.registerHooks({
    resolve(specifier, context, nextResolve) {
      try {
        return nextResolve(specifier, context);
      } catch (err) {
        return nextResolve(withJs(specifier, err), context);
      }
    },
  });
} else {
  // Node < 22.15: async hooks in a data: module
  const hooks = `export async function resolve(s, c, next) {
    try { return await next(s, c); } catch (e) {
      if (e?.code !== 'ERR_MODULE_NOT_FOUND' || !s.startsWith('.') || s.endsWith('.js')) throw e;
      return next(s + '.js', c);
    }
  }`;
  nodeModule.register(`data:text/javascript,${encodeURIComponent(hooks)}`);
}
