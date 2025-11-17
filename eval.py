import lighteval
from lighteval.logging.evaluation_tracker import EvaluationTracker
from lighteval.models.vllm.vllm_model import VLLMModel, VLLMModelConfig
from lighteval.pipeline import Pipeline, PipelineParameters, ParallelismManager
import argparse

aime25 = None


def main(model_name, thinking_mode = False):
    tracker = EvaluationTracker(
        "./res",
        hub_results_org = "carlesoctav"
    )
    tasks = (
      "aime25,math_500,gsm_plus,gpqa,ifeval,"
      "mixeval_hard:freeform,mixeval_hard:multichoice,"
      "mixeval_easy:freeform,mixeval_easy:multichoice"
    )
    p_params = PipelineParameters(ParallelismManager.VLLM)
    pipeline = Pipeline(
            tasks= tasks,
            pipeline_parameters = p_params,
            evaluation_tracker = tracker,
            model_config = VLLMModelConfig(
                model_name = model_name,
                max_model_length = 8192,
            )
    )


    pipeline.evaluate()
    pipeline.save_and_push_results()

    pipeline.show_results()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model_name",type = str, help = "name of model")
    parser.add_argument("--thinking_mode",type = bool, help = "enable thinking mode")
    args = parser.parse_args()
    main(args.model_name, thinking_mode = args.thinking_mode)


