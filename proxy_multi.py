import asyncio
import yaml
import uvicorn

from hacky_llama.service_multi import model_manager_app


if __name__ == "__main__":
    with open("config.yaml.multi") as f:
        config = yaml.safe_load(f)

    async def run_proxy(config):
        app = model_manager_app(config)
        uvicorn_config = uvicorn.Config(app=app, host="0.0.0.0", port=8000, log_level="debug")
        server = uvicorn.Server(uvicorn_config)
        await server.serve()
    asyncio.run(run_proxy(config))
