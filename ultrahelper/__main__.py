from .load import load_trainer, load_model, load_deployment_model
from .nn import InputSignatureWrap
from torch.fx import symbolic_trace
# from .test.pipeline_test import run_pipeline
from .pipeline import ParallelInferencePipeline
from .pipeline import *

import argparse

def get_args():
    parser = argparse.ArgumentParser(description="Pipeline control flags")
    parser.add_argument('--train', action='store_true', help='Runs training')
    parser.add_argument('--trace', action='store_true', help='Traces the modified model')
    parser.add_argument('--pipeline', action='store_true', help='Execute full parallel pipeline on infinite loop')
    return parser.parse_args()


def main():
    args = get_args()
    if args.train:
        trainer = load_trainer()
        trainer.train()
    elif args.trace:
        model = load_model()
        model = InputSignatureWrap(model)
        model_traced = symbolic_trace(model)
        model_traced.graph.print_tabular()
    
    # elif args.pipeline:
    #     raise NotImplementedError()
    elif args.pipeline:
        pipeline = ParallelInferencePipeline()
        pipeline.run()

    else:
         print("Please specify one of --train, --trace, or --pipeline")

    

if __name__ == '__main__':
    main()