from .core import process_directory
import argparse

def main():
    parser = argparse.ArgumentParser(description="AI Photo Culling Tool")
    parser.add_argument("folder", help="Folder containing images")
    parser.add_argument(
        "--profile",
        choices=["sports", "burlesque", "derby"],
        default="sports",
        help="Scoring profile",
    )
    args = parser.parse_args()
    process_directory(args.folder, args.profile)
