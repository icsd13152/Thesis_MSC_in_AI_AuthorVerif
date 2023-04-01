import click
import logging
import parse_data
import evaluate
import create_data_loader
import model
from pathlib import Path


logging.basicConfig(encoding='utf-8', level=logging.DEBUG)


def run_exp(input_data,output_data,models_path):
    logging.info("load Test data")
    corpus = parse_data.Corpus()
    orig_dict_data = corpus.parse_raw_data(input_data)
    data_loader = create_data_loader.getDataLoader(orig_dict_data)
    del corpus
    logging.info("load Models")
    modelEE,modelSI,modeGen = model.getModels(models_path)
    evaluator = evaluate.ContrastiveChunkerEvaluator(test_dataset=data_loader,
                                                     thresholdEE = 0.7, thresholdSI = 0.75,thresholdEE2 = 0.8,
                                                     thresholdSI2 = 0.8,thresholdGen = 0.6,outputPath =output_data)
    logging.info("Start Evaluation")
    evaluator.call(modelEE,modelSI,modeGen)
    logging.info("End Evaluation")
    del modelEE,modelSI,modeGen,evaluator

@click.option('-i', '--input-dataset-dir', type=click.Path(exists=True, file_okay=False),
              help="Path to the works.jsonl.")
@click.option('-o', '--output-dir', type=click.Path(exists=True, file_okay=False), default="./output",
              help="Path where to write the output file")
@click.option('-sEE', '--savepoint', type=click.Path(exists=True, file_okay=False), default="/software/models",
              help="Path to the saved model for Email-Essay with name model.pt")
@click.option('-sSI', '--savepoint', type=click.Path(exists=True, file_okay=False), default="/software/models",
              help="Path to the saved model for Speech-Interviews with name model.pt")
@click.option('-sGen', '--savepoint', type=click.Path(exists=True, file_okay=False), default="/software/models",
              help="Path to the Generic saved model with name model.pt")
@click.command()
def run(input_dataset_dir, output_dir, models_path):
    """
    $ python3 baseline-xgboost-runner.py \
        -i "<input-data-path>" \
        -o "<output-dir>" \
        -s "<saved-model-dir>"
    """

    run_exp(Path(input_dataset_dir), Path(output_dir), Path(models_path))

if __name__ == "__main__":
    run()

