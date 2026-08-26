Drop images here to test the trained model on pictures it has never seen.

    python -m src.predict --checkpoint outputs/baseline_pokenet/best.pt --dir test_images

Any of .jpg .jpeg .png .webp .bmp .gif works. Subfolders are searched too, so you
can organise by source if you like (test_images/artwork/, test_images/photos/).

Ignored by git - these are your scratch images, not part of the project.
