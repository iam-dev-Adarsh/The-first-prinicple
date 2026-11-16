PS F:\It's_AWESOME!\First_Principles\6.Decision_TreesSL> python main.py
Data: 120 points, max depth 4

Decision tree structure:
[y <= 4.81]  gain= 0.4375
    -> Predict: class 1 (56 samples)
    [y <= 5.51]  gain= 0.0339
        [x <= 7.70]  gain= 0.1587
            [y <= 5.10]  gain= 0.2755
                -> Predict: class 0 (4 samples)
                -> Predict: class 1 (3 samples)
            -> Predict: class 0 (5 samples)
        -> Predict: class 0 (52 samples)

Tree depth: 4
Training accuracy: 99.2%

Test predictions:
 (5, 7) -> class 0
 (5, 3) -> class 1
 (8, 5) -> class 0