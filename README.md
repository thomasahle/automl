The following command will search for CIFAR-10 models that maximize test accuracy, given a 20 second time budget:
```
python main.py cifar10 20 --device 0, --accelerator gpu --num-producers 3
```
The flag `--num-producers 3` means that three parallel LLM personalities will be proposing new programs to try.
Depending on the speed of your LLM and the time budget of your pytorch lightning model, you may want to use more or less than that.
