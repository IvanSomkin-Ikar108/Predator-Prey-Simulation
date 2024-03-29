import math
import random
import os
from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

from PIL import Image, ImageDraw, ImageFont


class Area:

    def __init__(
            self,
            net_pos: (int, int),
            img_pos: (int, int),
            predator: float = None,
            prey: float = None,
            prey_reproduction_rate_start: float = None,
            prey_reproduction_cycle_freq: float = None,
            predator_eating_rate: float = None,
            predator_dying_rate: float = None,
    ):
        self.id = None
        self.net_pos = net_pos
        self.img_pos = img_pos
        self.predator = (
            predator
            if predator is not None
            else random.uniform(1., 2.)
        )
        self.prey = (
            prey
            if prey is not None
            else random.uniform(1., 10.)
        )
        self.predator_next = self.predator
        self.prey_next = self.prey
        self.neighbors = []
        self.neighbors_2 = []
        self.prey_reproduction_rate_start = (
            prey_reproduction_rate_start
            if prey_reproduction_rate_start is not None
            else random.uniform(1., 5.)
        )
        self.prey_reproduction_cycle_freq = (
            prey_reproduction_cycle_freq
            if prey_reproduction_cycle_freq is not None
            else random.uniform(0.5, 1.) * 2 * np.pi
        )
        self.predator_eating_rate = (
            predator_eating_rate
            if predator_eating_rate is not None
            else random.uniform(0.5, 1.)
        )
        self.predator_dying_rate = (
            predator_dying_rate
            if predator_dying_rate is not None
            else random.uniform(1., 5.)
        )
        self.prey_history = [0, 0, 0, self.prey]
        self.predator_history = [0, 0, 0, self.predator]

    @staticmethod
    def hexagon_generator(img_x: int, img_y: int, hexagon_width: int):
        for angle in range(0, 360, 60):
            gen_x: int = img_x + \
                round(math.cos(math.radians(angle)) * hexagon_width // 2)
            gen_y: int = img_y + \
                round(math.sin(math.radians(angle)) * hexagon_width // 2)
            yield gen_x, gen_y

    def draw(self, draw: ImageDraw, hexagon_width: int):
        img_x, img_y = self.img_pos
        text_size = hexagon_width // 7
        hexagon = self.hexagon_generator(img_x, img_y, hexagon_width)
        draw.polygon(list(hexagon), outline='red', fill='white')

        text = f'id: {self.id}\nX: {round(self.prey, 2)}\nY: {round(self.predator, 2)}'
        text_offset_x = hexagon_width // 4
        text_offset_y = hexagon_width // 4
        draw.text((img_x - text_offset_x, img_y - text_offset_y), text=text, fill="black",
                  font=ImageFont.truetype("arial.ttf", size=text_size), align="left")

    def prey_reproduce_rate(self, t: float):
        return (
            self.prey_reproduction_rate_start *
            (1 + math.sin(self.prey_reproduction_cycle_freq * t))
        )

    def reproduce_interact_basic(self, time: float, time_step: float, data_precision: int):
        t = time
        x = self.prey
        y = self.predator
        a = self.prey_reproduce_rate(t)
        b = self.predator_eating_rate
        c = self.predator_dying_rate
        dt = time_step
        x_next = x * (1 + a * dt - y * dt)
        y_next = y * (1 + b * x * dt - c * dt)
        if x_next < 0:
            x_next = 0
        if y_next < 0:
            y_next = 0
        self.prey = round(x_next, data_precision)
        self.predator = round(y_next, data_precision)
        self.prey_history.pop(0)
        self.prey_history.append(self.prey)
        self.predator_history.pop(0)
        self.predator_history.append(self.predator)

    def reproduce_interact_pro(self, time: float, time_step: float, data_precision: int):
        t = time
        a = self.prey_reproduce_rate(t)
        b = self.predator_eating_rate
        c = self.predator_dying_rate
        dt = time_step
        x_im3 = self.prey_history[1]
        y_im3 = self.predator_history[1]
        x_im2 = self.prey_history[2]
        y_im2 = self.predator_history[2]
        x_im1 = self.prey
        y_im1 = self.predator
        x_i = (3 * x_im1 * dt * (a - y_im1) +
               (9 * x_im1 - 4.5*x_im2 + x_im3)) / 5.5
        y_i = (3 * y_im1 * dt * (b * x_im1 - c) +
               (9 * y_im1 - 4.5*y_im2 + y_im3)) / 5.5
        if x_i < 0:
            x_i = 0
        if y_i < 0:
            y_i = 0
        self.prey = round(x_i, data_precision)
        self.predator = round(y_i, data_precision)
        self.prey_history.pop(0)
        self.prey_history.append(self.prey)
        self.predator_history.pop(0)
        self.predator_history.append(self.predator)

    '''
    def migrate_next(self, time_step: float, data_precision: int,
                     predator_migration_rate: float = 0, prey_migration_rate: float = 0):
        predator_diff = sum([n.predator - self.predator for n in self.neighbors])
        prey_diff = sum([n.prey - self.prey for n in self.neighbors])
        self.predator_next = round(self.predator + predator_migration_rate * predator_diff, data_precision)
        self.prey_next = round(self.prey + prey_migration_rate * prey_diff, data_precision)
        self.t += time_step
    '''

    def initialize_neighbors(self, areas: []):
        self.neighbors.clear()
        neighbor_positions: [] = self.calculate_neighbor_positions()
        for a in areas:
            if a.net_pos in neighbor_positions:
                self.neighbors.append(a)
        return self.neighbors

    def calculate_neighbor_positions(self) -> []:
        net_x, net_y = self.net_pos
        return [
            (net_x, net_y - 1),
            (net_x, net_y + 1),
            (net_x - 1, net_y + net_x % 2 - 1),
            (net_x - 1, net_y + net_x % 2),
            (net_x + 1, net_y + net_x % 2 - 1),
            (net_x + 1, net_y + net_x % 2)
        ]

    def calculate_neighbours_2(self):
        self.neighbors_2.clear()
        for neighbor in self.neighbors:
            for neighbor_2 in neighbor.neighbors:
                if neighbor_2 is not self and neighbor_2 not in self.neighbors and neighbor_2 not in self.neighbors_2:
                    self.neighbors_2.append(neighbor_2)

    def __str__(self):
        return f'{self.predator}, {self.prey}, {self.net_pos}, {self.img_pos}, {[n.net_pos for n in self.neighbors]}'


class PredatorPreyRunResult:
    def __init__(self, area: Area, time_precision: int, data_precision: int):
        self.area_id = area.id
        self.predator0 = area.predator
        self.prey0 = area.prey
        self.a0 = area.prey_reproduction_rate_start
        self.b = area.predator_eating_rate
        self.c = area.predator_dying_rate
        self.w_coef = area.prey_reproduction_cycle_freq
        self.time_precision = time_precision
        self.data_precision = data_precision
        self.predator = []
        self.prey = []
        self.t = []
        self.plt = None
        self.fig_num = None

    def add_figure(self, p: plt, fig_num: int, prey_label: str = "Жертвы", predator_label: str = "Хищники"):
        self.plt = plt
        self.fig_num = fig_num
        t = self.t
        prey = self.prey
        predator = self.predator
        prey0 = self.prey0
        predator0 = self.predator0
        area_id = self.area_id
        time_precision = self.time_precision
        data_precision = self.data_precision
        a0 = self.a0
        b = self.b
        c = self.c
        w_coef = self.w_coef

        p.figure(fig_num)
        p.plot(t, prey, label=prey_label)
        p.plot(t, predator, label=predator_label)
        p.xticks(np.arange(0, max(t), max(t) / 20), rotation=90, fontsize=10)
        p.grid(True)
        p.legend()
        p.title(
            f'Area {area_id}\n' +
            f'(x0 = {round(prey0, 2)}, y0 = {round(predator0, 2)},' +
            f' a0 = {round(a0, 2)}, b = {round(b, 2)}, c = {round(c, 2)} w = {round(w_coef, 2)} * π)\n' +
            f'(Порядок точности: по времени - {time_precision}, по численности - {data_precision})',
            fontsize=10
        )
        p.xlabel('Время', fontsize=14)
        p.ylabel('Численность', fontsize=14)

    def add_plot(self, p: plt, fig_num: int, predator_label: str, prey_label: str):
        t = self.t

        p.figure(fig_num)
        p.plot(t, self.prey, label=prey_label)
        p.plot(t, self.predator, label=predator_label)
        p.legend()


class PredatorPreySimulation:
    def __init__(
            self,
            hexagon_width: int = 120,
            predator_migration_rate: float = 0.,
            prey_migration_rate: float = 0.,
    ):
        self.areas: [Area] = []
        self.output_directory: str = "./result"
        self.img_width = 0
        self.img_height = 0
        self.hexagon_width = hexagon_width
        self.hexagon_height: int = math.floor(
            self.hexagon_width * math.sin(math.pi / 3))
        self.hexagon_column_offset_y: int = self.hexagon_height // 2
        self.run_result = dict()
        self.predator_migration_rate = predator_migration_rate
        self.prey_migration_rate = prey_migration_rate
        self.predator_migration_matrix = None
        self.prey_migration_matrix = None
        self.steps = None
        self.time_precision = None
        self.data_precision = None
        self.render = None
        self.render_step_period = None

    def initialize_areas_with_image(self, img_path):
        with Image.open(img_path) as im:
            img_bw = im.convert("1")
            self.img_width, self.img_height = im.size
            net_measure_x = self.hexagon_width * 3 // 4
            net_measure_y = self.hexagon_height
            net_width = math.ceil(self.img_width / net_measure_x)
            net_height = math.ceil(self.img_height / net_measure_y)

            self.areas = []
            for net_x in range(0, net_width, 1):
                for net_y in range(0, net_height, 1):
                    img_x = net_x * net_measure_x
                    img_y = net_y * net_measure_y + \
                        self.hexagon_column_offset_y * (net_x % 2)
                    if (img_x < img_bw.width and img_y < img_bw.height
                            and self.is_pixel_black(img_bw, (img_x, img_y))):
                        self.areas.append(Area((net_x, net_y), (img_x, img_y)))

            for a in self.areas:
                a.initialize_neighbors(self.areas)
            for a in self.areas:
                a.calculate_neighbours_2()

    @staticmethod
    def is_pixel_black(img_bw, xy: (int, int)) -> bool:
        return img_bw.getpixel(xy) == 0

    def set_output_directory(self, output_directory):
        self.output_directory = output_directory

    def add_run_result(self, a: Area, t: int):
        self.run_result[a.id].predator.append(a.predator)
        self.run_result[a.id].prey.append(a.prey)
        self.run_result[a.id].t.append(t)

    def initialize_run_result(self, time_precision: int, data_precision: int):
        self.run_result = dict()
        for i in range(len(self.areas)):
            a = self.areas[i]
            a.id = i
            self.run_result[i] = PredatorPreyRunResult(
                a, time_precision, data_precision)

    def set_area_parameters(
            self,
            predator: float,
            prey: float,
            prey_reproduction_rate_start: float,
            prey_reproduction_cycle_freq: float,
            predator_eating_rate: float,
            predator_dying_rate: float,
            all_areas: bool,
            set_area_id: int
    ):
        if all_areas:
            a_list = self.areas
        else:
            a_list = [self.areas[set_area_id]]

        for a in a_list:
            a.predator = predator
            a.prey = prey
            a.prey_reproduction_rate_start = prey_reproduction_rate_start
            a.prey_reproduction_cycle_freq = prey_reproduction_cycle_freq
            a.predator_eating_rate = predator_eating_rate
            a.predator_dying_rate = predator_dying_rate

    def get_migration_matrix(self, migration_rate: float, time_step: float):
        matrix = np.zeros((len(self.areas), len(self.areas)),
                          int).astype('float')
        for area in self.areas:
            matrix[area.id][area.id] = 1. + float(len(area.neighbors)) * migration_rate * time_step +\
                float(len(area.neighbors_2)) * migration_rate * time_step
            for neighbour in area.neighbors:
                matrix[area.id][neighbour.id] = - \
                    1. * migration_rate * time_step
            for neighbour_2 in area.neighbors_2:
                matrix[area.id][neighbour_2.id] = - \
                    1. * migration_rate * time_step
        return matrix

    def initialize_migration_matrices(self, time_step: float):
        self.predator_migration_matrix = self.get_migration_matrix(
            self.predator_migration_rate, time_step)
        self.prey_migration_matrix = self.get_migration_matrix(
            self.prey_migration_rate, time_step)

    def migrate(self):
        prey_old = [a.prey for a in self.areas]
        predator_old = [a.predator for a in self.areas]
        prey_new = np.linalg.solve(self.prey_migration_matrix, prey_old)
        predator_new = np.linalg.solve(
            self.predator_migration_matrix, predator_old)
        for i in range(0, len(self.areas)):
            self.areas[i].prey = prey_new[i]
            self.areas[i].predator = predator_new[i]

    def render_step(self, img_path: str):
        area_img = Image.new(
            "RGB",
            (
                self.img_width + self.hexagon_width * 3 // 8,
                self.img_height + self.hexagon_height // 2
            ),
            (255, 255, 255)
        )
        draw = ImageDraw.Draw(area_img)
        for a in self.areas:
            a.draw(draw, self.hexagon_width)
        area_img.save(img_path)

    @staticmethod
    def get_condition_number(matrix):
        return np.linalg.cond(matrix, 2)

    def prepare_areas(self, time_step: float, data_precision: int):
        # 6 - это порядок (по 3 шага для точности по времени) точности по времени, прединициализируем начальные шаги
        # для разностной схемы из 4 точек
        time = 0
        for step in range(0, 6):
            if step % 2 == 0:
                for a in self.areas:
                    a.reproduce_interact_basic(time, time_step, data_precision)
            else:
                self.migrate()

            time += time_step
            for a in self.areas:
                self.add_run_result(a, time)
        return time

    def run(self, steps: int, time_precision: int, data_precision: int,
            render: bool, render_step_period: int):

        self.steps = steps
        self.time_precision = time_precision
        self.data_precision = data_precision
        self.render = render
        self.render_step_period = render_step_period

        time_step = math.pow(10, -time_precision)
        self.initialize_run_result(time_precision, data_precision)
        self.initialize_migration_matrices(time_step)

        self.print_migration_matrix_abstraction()

        time = self.prepare_areas(time_step, data_precision)

        for a in self.areas:
            self.add_run_result(a, time)

        prey_condition_number = self.get_condition_number(
            self.prey_migration_matrix)
        predator_condition_number = self.get_condition_number(
            self.predator_migration_matrix)

        print("Число обусловленности для миграции жертв : ", prey_condition_number)
        print("Число обусловленности для миграции хищников : ",
              predator_condition_number)

        run_directory = "result"

        if render:
            run_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            run_directory = os.path.join(
                self.output_directory, f'run_{run_datetime}')
            os.mkdir(run_directory)

        for step in range(0, steps):
            if step % 2 == 0:
                for a in self.areas:
                    a.reproduce_interact_pro(time, time_step, data_precision)
            else:
                self.migrate()

            time += time_step
            for a in self.areas:
                self.add_run_result(a, time)
            if render and step % render_step_period == 0:
                self.render_step(f'{run_directory}/step_{step}.jpg')

    def rerun(self):
        self.run(
            steps=self.steps,
            time_precision=self.time_precision,
            data_precision=self.data_precision,
            render=self.render,
            render_step_period=self.render_step_period
        )

    def get_run_result(self, area_id: int):
        return self.run_result[area_id]

    def print_migration_matrix_abstraction(self):
        # For TecX language
        # for area in self.areas:
        #     next_step: str = f'&\\left(1+{len(area.neighbors) + len(area.neighbors_2)}vdt\\right)x^{{n+1}}_{{{area.id}}}-vdt('
        #     for neighbour in area.neighbors:
        #         next_step += f'+x^{{n+1}}_{{{neighbour.id}}}'
        #     for neighbour_2 in area.neighbors_2:
        #         next_step += f'+x^{{n+2}}_{{{neighbour_2.id}}}'
        #     next_step += f')=x^{{n}}_{{{area.id}}}\\\\'
        #     print(next_step)

        for area in range(len(self.prey_migration_matrix)):
            neighbors = np.sort(np.nonzero(
                self.prey_migration_matrix[area]))[0]
            next_step = f'&\\left(1+{len(neighbors) - 1}vdt\\right)x^{{n+2}}_{{{area}}}'
            for neighbour in neighbors:
                if neighbour != area:
                    next_step += f'-vx^{{n+2}}_{{{neighbour}}}dt'
            next_step += f'=x^{{n+1}}_{{{area}}}\\\\'
            print(next_step)


if __name__ == '__main__':
    simulation = PredatorPreySimulation(
        120, predator_migration_rate=0.1, prey_migration_rate=0.1)
    simulation.initialize_areas_with_image('./assets/brazil.png')
    simulation.set_output_directory('result')

    # Сохраняем параметры запуска для последующего моделирования из тех же условий
    saved_areas: [Area] = deepcopy(simulation.areas)

    # Опция render, включает запись модели в виде изображения каждые render_step_period шагов моделирования
    simulation.run(steps=2500, time_precision=2, data_precision=3,
                   render=False, render_step_period=100)

    focused_areas = [30, 8, 32]

    # Строим базовые графики для выбранных областей
    simulation.get_run_result(30).add_figure(plt, 0)  # 2 neighbours
    simulation.get_run_result(8).add_figure(plt, 1)  # 4 neighbours
    simulation.get_run_result(32).add_figure(plt, 2)  # 6 neighbours

    # Строим общие графики для выбранных и областей и их соседей в схожих условиях смежности
    simulation.get_run_result(30).add_figure(
        plt, 3, "Жертвы - 30", "Хищники - 30")  # 2 neighbours
    simulation.get_run_result(21).add_figure(
        plt, 3, "Жертвы - 21", "Хищники - 21")  # 2 neighbours
    simulation.get_run_result(8).add_figure(
        plt, 4, "Жертвы - 8", "Хищники - 8")  # 4 neighbours
    simulation.get_run_result(13).add_figure(
        plt, 4, "Жертвы - 13", "Хищники - 13")  # 4 neighbours
    simulation.get_run_result(32).add_figure(
        plt, 5, "Жертвы - 32", "Хищники - 32")  # 6 neighbours
    simulation.get_run_result(33).add_figure(
        plt, 5, "Жертвы - 33", "Хищники - 33")  # 6 neighbours

    # Создаём базовые графики для сравнения значений после изменения условий моделирования
    simulation.get_run_result(30).add_figure(
        plt, 6, "Жертвы_def", "Хищники_def")  # 2 neighbours
    simulation.get_run_result(8).add_figure(
        plt, 7, "Жертвы_def", "Хищники_def")  # 4 neighbours
    simulation.get_run_result(32).add_figure(
        plt, 8, "Жертвы_def", "Хищники_def")  # 6 neighbours

    # Сохраняем первые результаты моделирования для последующего сравнения
    def_run_results: dict = {}
    for id in focused_areas:
        def_run_results[id] = simulation.get_run_result(id)

    # Задаём для симуляции такие же условия, как и при первом запуске
    simulation.areas = deepcopy(saved_areas)

    # Изменяем стартовые условия выбранных зон
    for id in focused_areas:
        simulation.set_area_parameters(
            predator=saved_areas[id].predator * 1.1,
            prey=saved_areas[id].prey * 1.1,
            prey_reproduction_rate_start=saved_areas[id].prey_reproduction_rate_start * 1.1,
            prey_reproduction_cycle_freq=saved_areas[id].prey_reproduction_cycle_freq,
            predator_eating_rate=min(
                saved_areas[id].predator_eating_rate * 1.1, 1.),
            predator_dying_rate=saved_areas[id].predator_dying_rate * 1.1,
            all_areas=False,
            set_area_id=id
        )

    # Запуск симуляции при заданных условиях
    simulation.rerun()

    # Добавляем на ранее созданные графики результаты моделирования после внесения изменений
    simulation.get_run_result(30).add_figure(
        plt, 6, "Жертвы_new", "Хищники_new")  # 2 neighbours
    simulation.get_run_result(8).add_figure(
        plt, 7, "Жертвы_new", "Хищники_new")  # 4 neighbours
    simulation.get_run_result(32).add_figure(
        plt, 8, "Жертвы_new", "Хищники_new")  # 6 neighbours

    #  Вычисляем среднее значение разницы для запуска с внесёнными изменениями
    for id in focused_areas:
        print(id, np.mean(np.array(simulation.get_run_result(id).predator) - np.array(def_run_results[id].predator)),
              np.mean(np.array(simulation.get_run_result(id).prey) - np.array(def_run_results[id].prey)))
    plt.show()
