import sys
import asyncio
import yaml
import uvicorn

from hacky_llama.service import model_manager_app


if __name__ == "__main__":
    with open("config.yaml") as f:
        config = yaml.safe_load(f)

    async def run_proxy(config):
        app = model_manager_app(config)
        uvicorn_config = uvicorn.Config(app=app, host="0.0.0.0", port=int(sys.argv[1]),
                                        log_level="debug", workers=2)
        server = uvicorn.Server(uvicorn_config)
        await server.serve()
    asyncio.run(run_proxy(config))
