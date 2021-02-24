from loguru import logger
import time
import argparse

def loguruInitialize(workdir: str):
    logger.add(workdir + '/pedestrian_attribute_recognition_{time}.log')

def buildArgParse():
    parse = argparse.ArgumentParser()
    parse.add_argument('--log_dir', default='.', type=str, required=False, help='Place to store log')
    return parse

if __name__ == "__main__":
    parser = buildArgParse()
    args = parser.parse_args()

    loguruInitialize(args.log_dir)
    logger.info("------------------- Main start ------------------\n")
