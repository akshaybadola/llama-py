import os
import json
import argparse
import asyncio
import uvicorn


from hacky_llama.gemma_service import create_app


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_root")
    parser.add_argument("--model_path")
    parser.add_argument("--lib_path")
    parser.add_argument("--mmproj_path")
    parser.add_argument("--n_predict", type=int)
    parser.add_argument("--overrides")
    parser.add_argument("--port", type=int)
    args = parser.parse_args()

    model_root = args.__dict__.pop("model_root")
    args.model_path = os.path.join(model_root, args.model_path)
    args.mmproj_path = os.path.join(model_root, args.mmproj_path)

    async def run_app(config):
        port = config.pop("port")
        config["overrides"] = json.loads(config["overrides"])
        app = await create_app(config)
        uvicorn_config = uvicorn.Config(app=app, host="0.0.0.0", port=port, log_level="debug")
        server = uvicorn.Server(uvicorn_config)
        await server.serve()
    asyncio.run(run_app(args.__dict__))
