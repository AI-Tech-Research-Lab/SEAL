import warnings
import sys
import os


warnings.simplefilter("ignore")

# Add the src/ dir to the path of import directory
sys.path.append(
    os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "src",
    )
)
