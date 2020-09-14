# Pytorch-Lightning-Templates

Set of training loops for solving common deep learning problems using PyTorch Lightning workflow. Each subproject contains examples of the data format that is expected. The project aims to make prototyping models as easy as possible while benefiting from all the "training tricks" that every deep learning practitioner should use.

For every subproject the training loop is defined in [train.py](base_scripts/base_train.py) file. Its structure is similar in each case: 

```python
checkpoint_callback = ModelCheckpoint(
        filepath=os.getcwd(),
        save_top_k=1,
        verbose=True,
        monitor='val_loss',
        mode='min',
    )

    TENSORBOARD_DIRECTORY = "logs/"
    EXPERIMENT_NAME = "experient"
    logger = TensorBoardLogger(TENSORBOARD_DIRECTORY, name=EXPERIMENT_NAME)

    #And then actual training
    trainer = Trainer(max_epochs=40,
                      logger=logger,
                      gpus=1,
                      # precision=16,
                      accumulate_grad_batches=4,
                      deterministic=True,
                      early_stop_callback=True,
                      checkpoint_callback=checkpoint_callback,
                      # resume_from_checkpoint = 'my_checkpoint.ckpt'
                      )

    trainer.fit(model, dm)
```
Every training loop automaticly loggs the loss and choosen metrics to the Tensorboard.


## Avalible templates

- [Image classification](image_classification/)
- [Multi label image classification](image_classification/)
- [Object detection](image_classification/)
- [Image Segmentation](image_classification/)


## Authors
* [Piotr Mazurek](https://github.com/tugot17)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details
