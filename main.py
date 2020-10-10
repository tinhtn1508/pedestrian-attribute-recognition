from loguru import logger

def loguruInitialize():
    logger.add("attribute-based-object-searching-for-surveillance-camera.log")

if __name__ == "__main__":
    loguruInitialize()
    logger.info("------------------- Main start ------------------\n")
