import asyncio
import yaml
import uvicorn

from hacky_llama.simple import create_app


if __name__ == "__main__":
    with open("config.yaml") as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)

    async def run_app(config):
        app = await create_app(config)
        uvicorn_config = uvicorn.Config(app=app, host="0.0.0.0", port=8000, log_level="debug")
        server = uvicorn.Server(uvicorn_config)
        await server.serve()
    asyncio.run(run_app(config))
