"""
DAMAGE - CV Pipeline for Warehouse Damage Detection
Main entry point
"""

from cv_pipeline.pipeline import Pipeline


def main():
    pipeline = Pipeline()
    pipeline.run()


if __name__ == "__main__":
    main()
