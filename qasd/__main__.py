import os

import uvicorn

if __name__ == "__main__":
    uvicorn.run("qasd.app:app", host=os.getenv("QASD_HOST", "0.0.0.0"), port=int(os.getenv("QASD_PORT", "8787")))
