import math
import random

from PIL import Image, ImageDraw, ImageFont

hexagon_width: int = 120
hexagon_height: int = int(hexagon_width * math.sin(math.pi / 3))
hexagon_offset_x: int = hexagon_width * 3 // 4
hexagon_offset_y: int = hexagon_height // 2


class Area:
    def __init__(self, net_position: (int, int), x: int, y: int):
        self.net_position = net_position
        self.x = x
        self.y = y
        self.neighbours = []
        self.predator = random.random() * 10.
        self.prey = random.random() * 100.

    def draw(self, draw: ImageDraw):
        hexagon = self.hexagon_generator(hexagon_width // 2)
        draw.polygon(list(hexagon), outline='red', fill='white')
        message = f'X: {round(self.prey, 2)}\nY: {round(self.predator, 2)}'
        draw.multiline_text((self.x - hexagon_width // 4, self.y - hexagon_height // 6), text=message, fill="black",
                            font=ImageFont.truetype("arial.ttf", size=16), align="left")

    def get_neighbours(self, areas: []):
        self.neighbours.clear()
        neighbours_positions: [] = self.get_neighbours_position()
        for area in areas:
            if area.net_position in neighbours_positions:
                self.neighbours.append(area)
        return self.neighbours

    def get_neighbours_position(self) -> []:
        net_x, net_y = self.net_position
        return [
            (net_x, net_y - 1),
            (net_x, net_y + 1),
            (net_x - 1, net_y + net_x % 2 - 1),
            (net_x - 1, net_y + net_x % 2),
            (net_x + 1, net_y + net_x % 2 - 1),
            (net_x + 1, net_y + net_x % 2)
        ]

    def hexagon_generator(self, edge_length):
        for angle in range(0, 360, 60):
            gen_x = self.x + math.cos(math.radians(angle)) * edge_length
            gen_y = self.y + math.sin(math.radians(angle)) * edge_length
            yield gen_x, gen_y

    def __str__(self):
        return f'{self.predator}, {self.prey}, {self.x}, {self.y}, {self.net_position}, {[n.net_position for n in self.neighbours]}'


def check_point(img: [], x: int, y: int) -> bool:
    return img[x, y] == (0, 0, 0, 255)


def get_areas(im: Image) -> [Area]:
    img = im.load()
    img_width, img_height = im.size
    areas: [Area] = []
    net_x, net_y = 0, 0

    for x in range(0, img_width - hexagon_offset_x, hexagon_width * 3 // 2):
        for y in range(0, img_height - hexagon_offset_y, hexagon_height):
            if check_point(img, x, y):
                areas.append(Area((net_x, net_y), x, y))
                img[x, y] = (255, 0, 0, 255)
            if check_point(img, x + hexagon_offset_x, y + hexagon_offset_y):
                areas.append(Area((net_x + 1, net_y), x + hexagon_offset_x, y + hexagon_offset_y))
                img[x + hexagon_offset_x, y + hexagon_offset_y] = (255, 0, 0, 255)
            net_y += 1
        net_x += 2
        net_y = 0

    for a in areas:
        a.get_neighbours(areas)

    return areas


if __name__ == '__main__':
    with Image.open("./assets/brazil.png") as image:
        areas: [Area] = get_areas(image)
        # image.show()
        simulation_step = (
            Image.new("RGB", (image.size[0] + 50, image.size[1] + 50), (255, 255, 255)), 0)  # Image + step number

        draw = ImageDraw.Draw(simulation_step[0])
        for area in areas:
            area.draw(draw)

        simulation_step[0].save(f'./result/simulation_step_{simulation_step[1]}.jpg')
        simulation_step[0].show()
