import numpy as np

def generate_fake_detections(num=50):
    data = []
    for i in range(num):
        data.append({
            "frame": i,
            "class": np.random.choice(["car","bike","bus","truck"]),
            "confidence": np.random.rand(),
            "xmin": np.random.randint(0,100),
            "ymin": np.random.randint(0,100),
            "xmax": np.random.randint(100,200),
            "ymax": np.random.randint(100,200)
        })
    return data
