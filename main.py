"""
Main Entry Point for Border Control FER System
----------------------------------------------
Author: Nazan Kafadaroglu
Date  : 2025-02-01

Description:
The entry point is this script, which first configures logging before starting the
graphical user interface (GUI) for facial expression detection in a border control
setup. `gui.py` contains the model loading mechanism and the actual GUI.

 - Code processes images locally and does not store personal data.
"""

import logging
from src.gui import launch_gui

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )

def main():
    """
    Main function for launching the Border Control FER system.
    """
    setup_logging()
    logging.info("=== Border Control Facial Expression Recognition System ===")

    # If you have a DB init step, uncomment or adjust as needed:
    # init_db()
    # logging.info("Database initialized (if not already).")

    # Now launch the GUI
    launch_gui()

if __name__ == "__main__":
    main()
