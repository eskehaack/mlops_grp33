import hydra
from torch.profiler import profile, ProfilerActivity
from torch.profiler import tensorboard_trace_handler
from MLops_project.train_model import main


@hydra.main(config_path="../MLops_project", config_name="config")
def profiler(cfg):
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        on_trace_ready=tensorboard_trace_handler("./tensorboard"),
    ) as prof:
        main(cfg)


if __name__ == "__main__":
    print("Remember to activate the tensorboard via visual studio code add on.")
    profiler()
