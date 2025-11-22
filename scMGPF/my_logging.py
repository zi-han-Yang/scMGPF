import logging

# Reset the log configuration to ensure the log level is INFO
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Get the logger object
logger = logging.getLogger(__name__)
