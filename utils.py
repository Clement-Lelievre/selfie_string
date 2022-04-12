import numpy as np

# bresenham's line algorithm
def bresenham(x0, y0, x1, y1) -> list:
    """
    Bresenham's line algorithm
    https://en.wikipedia.org/wiki/Bresenham%27s_line_algorithm
    """
    dx = abs(x1 - x0)
    sx = 1 if x0 < x1 else -1
    dy = abs(y1 - y0)
    sy = 1 if y0 < y1 else -1
    err = dx - dy
    xcoordinates = []
    ycoordinates = []
    while True:
        xcoordinates.append(x0)
        ycoordinates.append(y0)
        if x0 == x1 and y0 == y1:
            break
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x0 += sx
        if e2 < dx:
            err += dx
            y0 += sy
    return list(zip(xcoordinates, ycoordinates))

def best_pin(position: np.array, current_pin: tuple, pins: list = []) -> tuple:
    """From a position and a starting pin, computes and returns the highest-scoring destination pin"""
    destinations = [pin for pin in pins if pin[0] != current_pin[0] and abs(pins.index(current_pin) - pins.index(pin)) > 50]
    scores = []
    bresenhams = []
    for pin in destinations:
        #print(f'current pin {current_pin}, destination pin: {pin}')
        path_pixels = bresenham(*current_pin, *pin)
        bresenhams.append(path_pixels)
        #print(path_pixels)
        #print(max(path_pixels, key=lambda x : x[0]))
        score = sum([position[pixel] for pixel in path_pixels])
        scores.append(score)
    return destinations[scores.index(max(scores))], bresenhams[scores.index(max(scores))]