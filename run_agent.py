import argparse
import os
import sys
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from functools import reduce
from itertools import product
from math import factorial
from operator import mul

import cv2
import numpy as np
import pyautogui
from constraint import ExactSumConstraint, MaxSumConstraint, Problem
from PIL import ImageGrab
from pynput.keyboard import Key, Listener
from scipy.ndimage import binary_dilation, generate_binary_structure, label
from scipy.signal import convolve2d

# Global flags
start_agent = False
stop_agent = False
pause_agent = False
make_flag = False
fast_agent = False

# Level information
level_info = {
    1: ((9, 9), 10),
    2: ((11, 12), 19),
    3: ((13, 15), 32),
    4: ((14, 18), 47),
    5: ((16, 20), 66)
}

# Element name mapping
elem_name = {
    '0': 0,
    '1': 1,
    '2': 2,
    '3': 3,
    '4': 4,
    '5': 5,
    '6': 6,
    '7': 7,
    '8': 8,
    '9': 9,
    'mark': -1,
    'potion': -2,
    'clock': -3,
}

# Directories
cur_path = sys.argv[0]
cur_dir = os.path.dirname(cur_path)
resource_dir = os.path.join(cur_dir, 'resource')


def on_press(key):
    global start_agent
    global stop_agent
    global pause_agent
    global make_flag
    global fast_agent

    try:
        if key.char == 'b':
            start_agent = True
            print('Agent started.')
        elif key.char == 'p':
            pause_agent = not pause_agent
            print('Set pause to', pause_agent)
        elif key.char == 'f':
            make_flag = not make_flag
            print('Set flag to', make_flag)
        elif key.char == 't':
            fast_agent = not fast_agent
            print('Set turbo mode to', fast_agent)

    except AttributeError:
        if key == Key.esc:
            stop_agent = True
            print('Stop signal received.')


def get_image_array(image):
    image_array = np.array(image)
    image_array = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
    return image_array


def resize_image_by_resolution(image):
    # Get screen resolution
    screen_width, screen_height = ImageGrab.grab().size
    original_width, original_height = 2560, 1600

    image_height, image_width = image.shape[:2]
    width_ratio = screen_width / original_width
    height_ratio = screen_height / original_height
    resize_ratio = min(width_ratio, height_ratio)

    image_resized = cv2.resize(image, (int(image_width * resize_ratio),
                                       int(image_height * resize_ratio)), interpolation=cv2.INTER_AREA)
    return image_resized


def locate_image_on_screen(target_image_path, confidence=0.6):
    screenshot = pyautogui.screenshot()
    image_array = get_image_array(screenshot)
    target_image = cv2.imread(target_image_path)

    result = cv2.matchTemplate(image_array, target_image, cv2.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

    # print(target_image_path, 'confidence', max_val, confidence)

    if max_val >= confidence:
        top_left = max_loc
        h, w, _ = target_image.shape
        bottom_right = (top_left[0] + w, top_left[1] + h)
        return top_left, bottom_right
    else:
        return None


def load_elem_images(directory):
    images = {}
    for filename in os.listdir(directory):
        if filename.lower().endswith('.png'):
            name_key = filename.split('.')[0]
            if name_key not in elem_name:
                print(f'Unknown element name: {name_key}')
                continue
            name = elem_name[name_key]
            image_path = os.path.join(directory, filename)
            image = cv2.imread(image_path, cv2.IMREAD_COLOR)
            if image is None:
                print(f'Failed to load image: {image_path}')
                continue
            image = resize_image_by_resolution(image)
            images[name] = image
    return images


def preprocess_image(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)
    return blurred_image


def find_game_board():
    top_left_loc = locate_image_on_screen(
        os.path.join(resource_dir, 'topLeft.jpg'))
    bottom_right_loc = locate_image_on_screen(
        os.path.join(resource_dir, 'botRight.jpg'))

    # Return None when board corner not found
    if top_left_loc is None or bottom_right_loc is None:
        return None, None

    def get_middle(x):
        return (
            x[0][0] + (x[1][0] - x[0][0]) // 2,
            x[0][1] + (x[1][1] - x[0][1]) // 2
        )

    top_left = get_middle(top_left_loc)
    bottom_right = get_middle(bottom_right_loc)

    x, y = top_left
    w, h = bottom_right

    # Return None when board corner location is incorrectly identified
    if w - x < 0 or h - y < 0:
        return None, None

    print('Board top-left:', top_left)
    print('Board bottom-right:', bottom_right)

    return top_left, bottom_right


class ElementMatcher:
    def __init__(self, elem_images):
        self._elem_images = elem_images
        self._elem_info = {}

    def find_best_match_element(self, grid):
        best_match_score = float('inf')
        best_match_elem = None

        for elem_name, elem in self._elem_images.items():
            hist_score = self.calculate_histogram_similarity(
                elem_name, grid, elem)

            if hist_score < best_match_score:
                best_match_score = hist_score
                best_match_elem = elem_name

        return best_match_elem

    def calculate_histogram_similarity(self, elem_name, image1, image2):
        def crop_center(image, crop_size=(192, 192)):
            y, x = image.shape[:2]
            startx = x // 2 - (crop_size[1] // 2)
            starty = y // 2 - (crop_size[0] // 2)
            return image[starty:starty + crop_size[0], startx:startx + crop_size[1]]

        def replace_color_with_white(image, lower_bound, upper_bound):
            """Generalized function to replace a specific color range with white."""
            color_mask = cv2.inRange(image, lower_bound, upper_bound)
            image[color_mask > 0] = [255, 255, 255]  # Set to white in BGR
            return image

        def calculate_histogram(image):
            hist_hue = cv2.calcHist([image], [0], None, [256], [0, 256])
            hist_saturation = cv2.calcHist([image], [1], None, [256], [0, 256])
            hist_value = cv2.calcHist([image], [2], None, [256], [0, 256])

            cv2.normalize(hist_hue, hist_hue)
            cv2.normalize(hist_saturation, hist_saturation)
            cv2.normalize(hist_value, hist_value)

            return hist_hue, hist_saturation, hist_value

        # Compare histograms using Bhattacharyya distance
        def compare_histograms(hist1, hist2):
            return cv2.compareHist(hist1, hist2, cv2.HISTCMP_BHATTACHARYYA)

        lower_brown = np.array([25, 40, 60])
        upper_brown = np.array([50, 70, 100])
        lower_green = np.array([25, 70, 60])
        upper_green = np.array([60, 120, 110])

        if elem_name not in self._elem_info:
            if image2.shape[:2] != (256, 256):
                image2 = cv2.resize(image2, (256, 256))
                image2 = crop_center(image2)

            if elem_name in range(0, 9):
                image2 = replace_color_with_white(
                    image2, lower_brown, upper_brown)
            elif elem_name in [9, -1, -2, -3]:
                image2 = replace_color_with_white(
                    image2, lower_green, upper_green)

            hsv_image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2HSV)
            hist_image2_hue, hist_image2_saturation, hist_image2_value = calculate_histogram(
                hsv_image2)
            self._elem_info[elem_name] = (
                hist_image2_hue, hist_image2_saturation, hist_image2_value)

        # Resize images only if needed
        if image1.shape[:2] != (256, 256):
            image1 = cv2.resize(image1, (256, 256))
            image1 = crop_center(image1)

        # Apply color replacements based on element name
        if elem_name in range(0, 9):
            image1 = replace_color_with_white(image1, lower_brown, upper_brown)
        elif elem_name in [9, -1, -2, -3]:
            image1 = replace_color_with_white(image1, lower_green, upper_green)

        # Convert both images to HSV color space
        hsv_image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2HSV)

        # Calculate histograms for both images
        hist_image1_hue, hist_image1_saturation, hist_image1_value = calculate_histogram(
            hsv_image1)
        hist_image2_hue, hist_image2_saturation, hist_image2_value = self._elem_info[
            elem_name]

        # Compare histograms for each channel and calculate combined score
        hist_score_hue = compare_histograms(hist_image1_hue, hist_image2_hue)
        hist_score_saturation = compare_histograms(
            hist_image1_saturation, hist_image2_saturation)
        hist_score_value = compare_histograms(
            hist_image1_value, hist_image2_value)

        # Combine scores into a final similarity score
        hist_score = (hist_score_hue + hist_score_saturation +
                      hist_score_value) / 3.0

        return hist_score


class MineSweeperAgent:

    SPECIAL_ELEM = [-2, -3]

    def __init__(self, elem_images):
        self._element_matcher = ElementMatcher(elem_images)
        self.rows = None
        self.cols = None
        self.mines = None
        self._grid_height = None
        self._grid_width = None
        self._grid_location = {}
        self._top_left = None
        self._bottom_right = None
        self.finished = False
        self.prev_actions = []
        self.spell_cnt = 0
        self.guess_count = 0  # Lägg till en räknare för gissningar

    def initialize_board(self, top_left, bottom_right, level_num):
        size, self.mines = level_info[level_num]
        self.rows, self.cols = size
        self.finished = False
        self._grid_height = None
        self._grid_width = None
        self._grid_location = {}
        self._top_left = top_left
        self._bottom_right = bottom_right
        self.prev_spell_action = False

        self.board = np.zeros((self.rows, self.cols), dtype=np.int64)
        self.board_special = np.zeros((self.rows, self.cols), dtype=np.int64)
        self.prev_actions = []
        self.guess_count = 0  # Nollställ gissningsräknaren

        self.solver = Solver(self.cols, self.rows,
                             self.mines, stop_on_solution=False)

    def identify_game_board(self):
        x, y, w, h = self._top_left[0], self._top_left[1], self._bottom_right[0], self._bottom_right[1]
        board = pyautogui.screenshot(region=(x, y, w - x, h - y))
        board_array = get_image_array(board)
        return board_array

    def split_board_into_grids(self, board_array):
        height, width, _ = board_array.shape
        # Initialize grid location on the screen
        if self._grid_height is None:
            self._grid_height = height / self.rows
            self._grid_width = width / self.cols
            for i in range(self.rows):
                for j in range(self.cols):
                    self._grid_location[(i, j)] = (self._top_left[0] + j * self._grid_width + self._grid_width / 2,
                                                   self._top_left[1] + i * self._grid_height + self._grid_height / 2)

        grids = []
        for i in range(self.rows):
            for j in range(self.cols):
                start_x, end_x = int(
                    j * self._grid_width), int((j + 1) * self._grid_width)
                start_y, end_y = int(
                    i * self._grid_height), int((i + 1) * self._grid_height)
                grid = board_array[start_y:end_y, start_x:end_x]
                grids.append(grid)
        return grids

    def update_elements(self, grids):
        def assign_elem(x, y, elem):
            if self.board[x, y] == -1:
                return

            if elem in self.SPECIAL_ELEM:
                self.board[x, y] = 9
                self.board_special[x, y] = elem
            else:
                self.board[x, y] = elem

        if not self.prev_actions:
            for i, grid in enumerate(grids):
                x, y = i // self.cols, i % self.cols
                elem = self._element_matcher.find_best_match_element(grid)
                assign_elem(x, y, elem)
        else:
            visited = set(self.prev_actions)
            check_list = deque(self.prev_actions)

            # BFS
            while check_list:
                x, y = check_list.popleft()
                i = y + x * self.cols
                if i >= len(grids):
                    continue
                grid = grids[i]
                new_elem = self._element_matcher.find_best_match_element(grid)

                if new_elem != self.board[x, y]:
                    assign_elem(x, y, new_elem)
                    for neighbor in self.get_neighbors(x, y):
                        if neighbor not in visited and 0 <= neighbor[0] < self.rows and 0 <= neighbor[1] < self.cols:
                            check_list.append(neighbor)
                            visited.add(neighbor)

        # Optimera sättet att hitta speciala index
        special_indices = np.argwhere(self.board_special != 0)
        self.special_indices = [tuple(idx) for idx in special_indices]

    def get_neighbors(self, x, y):
        # Förhindra att generera tuples i loopar
        neighbors = [
            (nx, ny)
            for nx in range(max(0, x - 1), min(self.rows, x + 2))
            for ny in range(max(0, y - 1), min(self.cols, y + 2))
            if not (nx == x and ny == y)
        ]
        return neighbors

    def convert_board(self):
        # Effektivare konvertering med NumPy
        state = self.board.copy().astype(float)
        state[self.board == -1] = np.nan  # Unrevealed
        state[self.board == 9] = np.nan  # Unrevealed (special)
        state = state.tolist()
        return state

    def get_action(self):
        state = self.convert_board()
        prob = self.solver.solve(state)

        # Logga sannolikheterna
        print("Current Probabilities:")
        print(prob)

        if np.isnan(prob).all() or (self.remaining_mines() == 0 and self.prev_spell_action):
            self.finished = True
            print("Finished level.")
            return None, None, None

        self.prev_spell_action = False

        flag_actions = []
        reveal_actions = []
        spell_actions = []

        # Flag actions: cells med sannolikhet 1
        flag_mask = (prob == 1) & (self.board != "flag")
        flagged_indices = np.argwhere(flag_mask)
        for idx in flagged_indices:
            y, x = idx
            self.board[y, x] = -1
            flag_actions.append((y, x))
            print(f"Flagging mine at ({y}, {x})")

        # Best prob: minimal sannolikhet
        best_prob = np.nanmin(prob)
        best_mask = (prob == best_prob)
        ys, xs = np.where(best_mask)

        print(f"Best probability: {best_prob}")

        # Endast avslöja om best_prob är 0
        if best_prob == 0:
            for y, x in zip(ys, xs):
                reveal_actions.append((y, x))
                print(f"Revealing safe cell at ({y}, {x})")
        else:
            if self.spell_cnt <= 5 and not self.is_mana_empty():
                # Använd spell snarare än att gissa
                spell_actions = list(zip(*np.where(self.board == 9)))
                self.spell_cnt += 1
                self.prev_spell_action = True
                print(f"Using spell on cells: {spell_actions}")
            else:
                # Gissa: använd policy
                x_click, y_click = self.solver.corner_then_edge2_policy(prob)
                reveal_actions.append((y_click, x_click))
                self.guess_count += 1  # Öka gissningsräknaren
                print(f"Guessing and revealing cell at ({y_click}, {x_click})")

        # Edge case: Om alla grannar flaggas, avslöja cellen
        for x in range(self.rows):
            for y in range(self.cols):
                if self.board[x, y] == 9:
                    all_flagged = all(
                        self.board[i, j] == -1 for i, j in self.get_neighbors(x, y)
                        if 0 <= i < self.rows and 0 <= j < self.cols
                    )
                    if all_flagged and (x, y) not in reveal_actions:
                        reveal_actions.append((x, y))
                        print(
                            f"Edge case: Revealing cell at ({x}, {y}) because all neighbors are flagged")

        if reveal_actions and spell_actions:
            best_spell_action = self.get_best_spell_action(spell_actions)
            spell_actions = [best_spell_action]
            print(f"Best spell action selected: {best_spell_action}")

        self.prev_actions = flag_actions + reveal_actions + spell_actions

        return flag_actions, reveal_actions, spell_actions

    def remaining_mines(self):
        flagged_mines_count = np.sum(self.board == -1)
        remaining_mines = self.mines - flagged_mines_count
        return remaining_mines

    def is_mana_empty(self):
        loc_1 = locate_image_on_screen(os.path.join(
            resource_dir, 'mana_1.jpg'), confidence=0.8)
        loc_2 = locate_image_on_screen(os.path.join(
            resource_dir, 'mana_2.jpg'), confidence=0.8)
        # Return True if both loc_1 and loc_2 are None, meaning no mana
        mana_status = loc_1 is None and loc_2 is None
        print(f"Mana status: {'Empty' if mana_status else 'Available'}")
        return mana_status

    def get_best_spell_action(self, spell_actions):
        max_info_gain = 0
        best_index = spell_actions[0] if spell_actions else None

        for i, j in spell_actions:
            neighbors = 0
            mines = 0
            value = 0

            for x, y in self.get_neighbors(i, j):
                if 0 <= x < self.rows and 0 <= y < self.cols:
                    if self.board[x, y] in range(1, 9):
                        neighbors += 1
                    elif self.board[x, y] == -1:
                        mines += 1

            value += neighbors - mines
            value += neighbors * 2

            if value > max_info_gain:
                max_info_gain = value
                best_index = (i, j)

        return best_index

    def take_action(self, flag_actions, reveal_actions, spell_actions):
        actions = []

        # Batch flag actions
        for index in flag_actions:
            x, y = self._grid_location[index]
            if make_flag:
                actions.append(('right_click', (x, y)))

        # Batch reveal actions
        for index in reveal_actions:
            x, y = self._grid_location[index]
            actions.append(('click', (x, y)))

        # Batch spell actions only if mana is not empty
        if spell_actions and not self.is_mana_empty():
            for index in spell_actions:
                x, y = self._grid_location[index]
                actions.append(('press', '1'))
                actions.append(('click', (x, y)))
                actions.append(
                    ('move', (self._bottom_right[0] + 5, self._bottom_right[1] + 5)))
            print(f"Spell actions queued: {spell_actions}")
        elif spell_actions and self.is_mana_empty():
            print("Mana is empty. Skipping spell actions.")

        # Utför alla åtgärder i batch
        for action, params in actions:
            if action == 'right_click':
                pyautogui.moveTo(params[0], params[1],
                                 duration=0.05)  # Snabbare rörelse
                pyautogui.rightClick()
            elif action == 'click':
                pyautogui.moveTo(params[0], params[1], duration=0.05)
                pyautogui.click()
            elif action == 'press':
                pyautogui.press(params)
            elif action == 'move':
                pyautogui.moveTo(params[0], params[1], duration=0.05)

        # Hantera spell-sömn endast om spell-åtgärder utförs
        if spell_actions and not self.is_mana_empty():
            sleep_duration = 2 if fast_agent else 4
            print(
                f"Sleeping for {sleep_duration} seconds after spell actions.")
            time.sleep(sleep_duration)

        # Flytta musen till botten höger för att undvika interferens
        pyautogui.moveTo(
            self._bottom_right[0] + 5, self._bottom_right[1] + 5, duration=0.05)


def dilate(bool_ar):
    return binary_dilation(bool_ar, structure=generate_binary_structure(2, 2))


def neighbors(bool_ar):
    return bool_ar ^ dilate(bool_ar)


def neighbors_xy(x, y, shape):
    return neighbors(mask_xy(x, y, shape))


def mask_xy(x, y, shape):
    mask = np.zeros(shape, dtype=bool)
    if 0 <= y < shape[0] and 0 <= x < shape[1]:
        mask[y, x] = True
    return mask


def boundary(state):
    return neighbors(~np.isnan(state))


def count_neighbors(bool_ar):
    filter = np.ones((3, 3))
    filter[1, 1] = 0
    return convolve2d(bool_ar, filter, mode='same')


def reduce_numbers(state, mines=None):
    num_neighboring_mines = count_neighbors(mines)
    state[~np.isnan(state)] -= num_neighboring_mines[~np.isnan(state)]
    return state


def no_preference(prob):
    return prob == np.nanmin(prob)


def edges(prob):
    selection = np.zeros(prob.shape, dtype=bool)
    selection[[0, prob.shape[0]-1], 1:prob.shape[1]-1] = True
    selection[1:prob.shape[0]-1, [0, prob.shape[1]-1]] = True
    return selection


def corners(prob):
    selection = np.zeros(prob.shape, dtype=bool)
    selection[[0, 0, prob.shape[0]-1, prob.shape[0]-1],
              [0, prob.shape[1]-1, 0, prob.shape[1]-1]] = True
    return selection


def corners2(prob):
    selection = np.zeros(prob.shape, dtype=bool)
    selection[[1, 1, prob.shape[0]-2, prob.shape[0]-2],
              [1, prob.shape[1]-2, 1, prob.shape[1]-2]] = True
    return selection


def edges2(prob):
    selection = np.zeros(prob.shape, dtype=bool)
    selection[[1, prob.shape[0]-2], 2:prob.shape[1]-2] = True
    selection[2:prob.shape[0]-2, [1, prob.shape[1]-2]] = True
    return selection


def nearest(preferred, prob):
    if np.isnan(prob).any():
        search_mask = binary_dilation(np.isnan(prob))
        while not (search_mask & preferred).any():
            search_mask = binary_dilation(search_mask)
        return search_mask & preferred
    else:
        return preferred


def random_selection(preferred, prob):
    from random import randrange
    """ Select a random square. """
    ys, xs = preferred.nonzero()
    if len(xs) == 0:
        return preferred
    idx = randrange(len(xs))
    x, y = xs[idx], ys[idx]
    selected = np.zeros(preferred.shape, dtype=bool)
    selected[y, x] = True
    return selected


def inward_corner(preferred, prob):
    search_mask = np.zeros(preferred.shape, dtype=bool)
    search_mask[[0, 0, search_mask.shape[0]-1, search_mask.shape[0]-1],
                [0, search_mask.shape[1]-1, 0, search_mask.shape[1]-1]] = True

    while not (search_mask & preferred).any():
        search_mask = binary_dilation(search_mask)
    return search_mask & preferred


def make_policy(preferences=no_preference, selection_methods=nearest):
    def template(preferences, selection_methods, prob):
        best = prob == np.nanmin(prob)
        for preference_selector in preferences:
            preferred = preference_selector(prob)
            # Now only look at those that have optimal probabilities.
            preferred = preferred & best
            if preferred.any():
                for selection_method in selection_methods:
                    preferred = selection_method(preferred, prob)
                    ys, xs = preferred.nonzero()
                    if len(xs) == 1:
                        return xs[0], ys[0]
        # Fall back to random selection
        return random_selection(best, prob)

    if not isinstance(preferences, list):
        preferences = [preferences]
    if not isinstance(selection_methods, list):
        selection_methods = [selection_methods]
    preferences.append(no_preference)
    selection_methods.append(random_selection)
    return lambda prob: template(preferences, selection_methods, prob)


class Solver:
    def __init__(self, width, height, total_mines, stop_on_solution=True):

        self._total_mines = total_mines
        self.known = np.full((height, width), np.nan, dtype=np.float32)
        self._stop_on_solution = stop_on_solution
        self.corner_then_edge2_policy = make_policy(
            [corners2, edges2], [nearest, inward_corner])

    def known_mine_count(self):
        return np.sum(self.known == 1, dtype=int)

    def mines_left(self):
        return self._total_mines - self.known_mine_count()

    def solve(self, state):
        state = np.array(
            [[state[y][x] if isinstance(state[y][x], int) else np.nan for x in range(
                len(state[0]))] for y in range(len(state))]
        )

        if not np.isnan(state).all():
            self.known[~np.isnan(state)] = 0
            prob, state = self._counting_step(state)
            if self._stop_on_solution and not np.isnan(prob).any() and 0 in prob:
                return prob
            prob = self._cp_step(state, prob)
            # Validera sannolikheterna
            assert np.all(prob >= 0) and np.all(
                prob <= 1), "Invalid probabilities detected!"
            # Logga sannolikheterna för debugging
            print("Solver probabilities:")
            print(prob)
            return prob
        else:
            return np.full(state.shape, self._total_mines / state.size, dtype=np.float32)

    def _counting_step(self, state):
        result = np.full(state.shape, np.nan, dtype=np.float32)
        new_results = True
        state = reduce_numbers(state, self.known == 1)
        unknown_squares = np.isnan(state) & np.isnan(self.known)

        while new_results:
            num_unknown_neighbors = count_neighbors(unknown_squares)
            solutions = (state == num_unknown_neighbors) & (
                num_unknown_neighbors > 0)
            known_mines = unknown_squares & reduce(np.logical_or,
                                                   [neighbors_xy(x, y, state.shape) for y, x in zip(
                                                       *solutions.nonzero())],
                                                   np.zeros(state.shape, dtype=bool))
            self.known[known_mines] = 1
            state = reduce_numbers(state, known_mines)
            unknown_squares = unknown_squares & ~known_mines
            num_unknown_neighbors = count_neighbors(unknown_squares)

            solutions = (state == 0) & (num_unknown_neighbors > 0)
            known_safe = unknown_squares & reduce(np.logical_or,
                                                  [neighbors_xy(x, y, state.shape) for y, x in zip(
                                                      *solutions.nonzero())],
                                                  np.zeros(state.shape, dtype=bool))
            self.known[known_safe] = 0
            unknown_squares = unknown_squares & ~known_safe

            result[known_safe] = 0
            result[known_mines] = 1
            new_results = (known_safe | known_mines).any()

        return result, state

    def _cp_step(self, state, prob):
        components, num_components = self._components(state)
        c_counts = []
        c_probs = []
        m_known = self.known_mine_count()

        for c in range(1, num_components + 1):
            areas, constraints = self._get_areas(state, components == c)
            problem = Problem()
            for v in areas.values():
                problem.addVariable(v, range(len(v) + 1))
            for constraint in constraints:
                problem.addConstraint(
                    constraint, [v for k, v in areas.items() if constraint in k])
            problem.addConstraint(MaxSumConstraint(
                self._total_mines - m_known), list(areas.values()))
            solutions = problem.getSolutions()

            model_count_by_m = {}  # {m: #models}
            model_prob_by_m = {}  # {m: prob of the average component model}
            for solution in solutions:
                m = sum(solution.values())
                model_count = self._count_models(solution)
                model_count_by_m[m] = model_count_by_m.get(m, 0) + model_count
                model_prob = np.zeros(prob.shape, dtype=np.float32)
                for area, m_area in solution.items():
                    model_prob[tuple(zip(*area))] = m_area / len(area)
                model_prob_by_m[m] = model_prob_by_m.get(
                    m, np.zeros(prob.shape, dtype=np.float32)) + model_count * model_prob
            # Normalize probabilities
            model_prob_by_m = {
                m: model_prob / model_count_by_m[m] for m, model_prob in model_prob_by_m.items()
            }
            c_probs.append(model_prob_by_m)
            c_counts.append(model_count_by_m)

        prob = self._combine_components(state, prob, c_probs, c_counts)

        return prob

    def _combine_components(self, state, prob, c_probs, c_counts):
        # Skapa maskar för lösningsområden
        solution_mask = boundary(state) & np.isnan(self.known)
        unconstrained_squares = np.isnan(
            state) & ~solution_mask & np.isnan(self.known)
        n = unconstrained_squares.sum(dtype=int)  # Definiera 'n' här

        # Debugging: Inspektera värdet av 'n' och 'unconstrained_squares'
        print(f"Antal okonstrainta rutor (n): {n}")
        print(f"Unconstrained squares mask:\n{unconstrained_squares}")

        if c_probs:
            try:
                # Se till att d inte är tomt
                min_mines = sum([min(d) for d in c_probs if d])
                max_mines = sum([max(d) for d in c_probs if d])
            except ValueError as e:
                print(f"Fel vid beräkning av min/max mines: {e}")
                min_mines = 0
                max_mines = 0

            mines_left = self.mines_left()
            weights = self._relative_weights(
                range(min_mines, min(max_mines, mines_left) + 1), n)
            total_weight = 0  # Vikten av kombinerade lösningar
            # Summan av viktade sannolikheter
            total_prob = np.zeros(prob.shape, dtype=np.float32)

            for c_ms in product(*[d.keys() for d in c_probs if d]):
                m = sum(c_ms)
                try:
                    comb_prob = reduce(
                        np.add, [c_probs[c][c_m] for c, c_m in enumerate(c_ms)]
                    )
                    comb_model_count = reduce(
                        mul, [c_counts[c][c_m] for c, c_m in enumerate(c_ms)]
                    )
                    weight = weights[m] * comb_model_count
                    total_weight += weight
                    total_prob += weight * comb_prob
                except Exception as e:
                    print(f"Fel vid kombination av c_probs: {e}")
                    continue

            if total_weight > 0:
                total_prob /= total_weight
                prob[solution_mask] = total_prob[solution_mask]
            else:
                print("Total weight är 0, inga viktade sannolikheter tillgängliga.")

        if n > 0:
            m_known = self.known_mine_count()
            remaining_mines = self._total_mines - m_known - \
                prob[~np.isnan(prob) & np.isnan(self.known)].sum()
            if n != 0:
                prob[unconstrained_squares] = remaining_mines / n
            else:
                print("n är 0, kan inte dela med 0.")

        certain_mask = np.isnan(self.known) & ((prob == 0) | (prob == 1))
        self.known[certain_mask] = prob[certain_mask]
        return prob

    def _count_models(self, solution):
        return reduce(mul, [self.combinations(len(area), m) for area, m in solution.items()], 1)

    def _components(self, state):
        numbers_mask = dilate(np.isnan(state) & np.isnan(
            self.known)) & ~np.isnan(state)
        labeled, num_components = label(numbers_mask)
        number_boundary_masks = [neighbors(labeled == c) & np.isnan(self.known) & np.isnan(state) for c in
                                 range(1, num_components + 1)]
        i = 0
        while i < len(number_boundary_masks) - 1:
            j = i + 1
            while j < len(number_boundary_masks):
                if (number_boundary_masks[i] & number_boundary_masks[j]).any():
                    number_boundary_masks[i] = number_boundary_masks[i] | number_boundary_masks[j]
                    del number_boundary_masks[j]
                    i -= 1
                    break
                j += 1
            i += 1

        labeled = np.zeros(state.shape, dtype=int)
        num_components = len(number_boundary_masks)
        for c, mask in enumerate(number_boundary_masks, 1):
            labeled[mask] = c

        i = 1
        while i <= num_components - 1:
            j = i + 1
            while j <= num_components:
                if not np.isnan(state[dilate(labeled == i) & dilate(labeled == j)]).all():
                    labeled[labeled == j] = i
                    labeled[labeled > j] -= 1
                    num_components -= 1
                    i -= 1
                    break
                j += 1
            i += 1
        return labeled, num_components

    @staticmethod
    def _get_areas(state, mask):
        constraints_mask = neighbors(mask) & ~np.isnan(state)
        constraint_list = [ExactSumConstraint(
            int(num)) for num in state[constraints_mask]]
        constraints = np.full(state.shape, None, dtype=object)
        constraints[constraints_mask] = constraint_list
        applied_constraints = np.empty(state.shape, dtype=object)
        for y, x in zip(*mask.nonzero()):
            applied_constraints[y, x] = []
        for yi, xi in zip(*constraints_mask.nonzero()):
            constrained_mask = neighbors_xy(xi, yi, mask.shape) & mask
            for yj, xj in zip(*constrained_mask.nonzero()):
                applied_constraints[yj, xj].append(constraints[yi, xi])

        mapping = {}
        for yi, xi in zip(*mask.nonzero()):
            # Convert to tuple, so we can use it as a hash key.
            k = tuple(applied_constraints[yi, xi])
            if k not in mapping:
                mapping[k] = []
            mapping[k].append((yi, xi))
        mapping = {k: tuple(v) for k, v in mapping.items()}
        return mapping, constraint_list

    @staticmethod
    def combinations(n, m):
        return factorial(n) // (factorial(n - m) * factorial(m)) if 0 <= m <= n else 0

    def _relative_weights(self, ms_solution, n):
        mines_left = self.mines_left()
        if n == 0:
            return {m: 1 for m in ms_solution}
        weights = {}
        for m_next_solution in ms_solution:
            if m_next_solution == 0:
                weights[m_next_solution] = 1 / n if n != 0 else 0
            else:
                weights[m_next_solution] = 1 / \
                    m_next_solution if m_next_solution != 0 else 0
        return weights


def test_case():
    top_left = (750, 424)
    bottom_right = (1809, 1271)

    agent = MineSweeperAgent(None)
    agent.initialize_board(top_left, bottom_right, 5)

    x = np.array(
        [
            [0, 0, 0, 0, 0, 0, 1, -1, 2, 2, -1, 2, 2, -1, 1, 0, 1, -1, -1, 1],
            [0, 0, 0, 0, 0, 0, 1, 2, 3, -1, 2, 2, -1, 2, 1, 0, 1, 2, 2, 1],
            [1, 1, 1, 0, 1, 1, 2, 2, -1, 2, 1, 1, 1, 1, 0, 0, 1, 1, 1, 0],
            [1, -1, 1, 0, 1, -1, 2, -1, 2, 2, 1, 1, 0, 1, 1, 1, 1, -1, 1, 0],
            [2, 2, 2, 0, 1, 1, 3, 3, 3, 2, -1, 2, 1, 2, -1, 2, 2, 2, 1, 0],
            [2, -1, 1, 0, 0, 0, 1, -1, -1, 3, 3, 4, -1, 3, 1, 2, -1, 2, 2, 2],
            [-1, 2, 1, 0, 0, 0, 1, 2, 3, -1, 3, -1, -1, 3, 2, 3, 2, 3, -1, -1],
            [1, 1, 1, 2, 3, 2, 1, 0, 1, 1, 3, -1, 3, 2, -1, -1, 3, 4, -1, -1],
            [1, 0, 1, -1, -1, -1, 1, 0, 0, 0, 1, 1, 1, 1, 3, -1, 3, -1, -1, 9],
            [0, 0, 2, 4, -1, 5, 3, 1, 0, 0, 0, 0, 0, 1, 2, 2, 2, 3, 5, 9],
            [1, 1, 1, -1, 3, -1, -1, 2, 0, 1, 1, 1, 0, 1, -1, 1, 0, 1, -1, -1],
            [-1, 1, 2, 2, 4, 4, -1, 3, 1, 1, -1, 2, 1, 2, 2, 2, 2, 2, 5, -1],
            [2, 2, 1, -1, 3, -1, 3, -1, 1, 1, 1, 2, -1, 2, 2, -1, 2, -1, 3, -1],
            [-1, 1, 1, 2, -1, 2, 2, 1, 2, 1, 1, 1, 2, -1, 3, 2, 3, 2, 3, 2],
            [2, 2, 1, 1, 1, 1, 0, 0, 2, -1, 2, 0, 1, 1, 3, -1, 2, 2, -1, 2],
            [1, -1, 1, 0, 0, 0, 0, 0, 2, -1, 2, 0, 0, 0, 2, -1, 2, 2, -1, 2]
        ]
    )
    print("Test Board:")
    print(x)

    agent.board = x
    flag_actions, reveal_actions, spell_actions = agent.get_action()

    print('flag_actions', flag_actions)
    print('reveal_actions', reveal_actions)
    print('spell_actions', spell_actions)


def run():
    elem_images = load_elem_images(resource_dir)
    agent = MineSweeperAgent(elem_images)

    global start_agent
    global stop_agent
    level_num = 1

    while True:
        while not start_agent and not stop_agent:
            time.sleep(1)

        if stop_agent:
            print("Program stopped by user.")
            time.sleep(3)
            sys.exit()

        pyautogui.click(interval=0.2)
        print('Start level', level_num)

        # Locate game board on screen
        while True:
            if stop_agent:
                print("Program stopped by user.")
                time.sleep(3)
                sys.exit()

            top_left, bottom_right = find_game_board()
            if top_left is None or bottom_right is None:
                print(
                    'Cannot locate the game board, make sure the game is on the screen.')
                time.sleep(5)
            else:
                break

        x, y = top_left
        w, h = bottom_right
        agent.initialize_board(top_left, bottom_right, level_num)

        # Klicka på startpositionen
        pyautogui.moveTo(x + (w - x) // 5, y + (h - y) // 5, duration=0.1)
        pyautogui.click(interval=0.5)
        pyautogui.moveTo(w + 5, h + 5, duration=0.1)

        while True:
            if stop_agent:
                print("Program stopped by user.")
                time.sleep(3)
                sys.exit()

            if pause_agent:
                time.sleep(0.5)
                continue

            t1 = time.time()
            if not fast_agent:
                # Double check
                board1 = None
                board2 = None
                while board1 is None or not np.array_equal(board1, board2):
                    t = time.time()
                    board_array = agent.identify_game_board()
                    grids = agent.split_board_into_grids(board_array)
                    agent.update_elements(grids)
                    board1 = agent.board.copy()
                    cost = time.time() - t1
                    time.sleep(max(1.0 - cost, 0))

                    board_array = agent.identify_game_board()
                    grids = agent.split_board_into_grids(board_array)
                    agent.update_elements(grids)
                    board2 = agent.board.copy()
            else:
                board_array = agent.identify_game_board()
                grids = agent.split_board_into_grids(board_array)
                agent.update_elements(grids)

            print(agent.board)
            print()

            # Uppdatera gridor igen efter avslöjanden
            flag_actions, reveal_actions, spell_actions = agent.get_action()

            if agent.finished:
                print('Finish level', level_num)

                if level_num == 5:
                    sys.exit()
                else:
                    print(
                        "Move the mouse over the 'Continue' button and press 'b' to start")

                start_agent = False
                break

            print()
            print('flag_actions', flag_actions)
            print('reveal_actions', reveal_actions)
            print('spell_actions', spell_actions)

            agent.take_action(flag_actions, reveal_actions, spell_actions)
            print('Time cost:', time.time() - t1)

            print()
            print('=' * 50)
            print()

        level_num += 1

        if stop_agent:
            break


if __name__ == "__main__":
    # Uncomment to run test case
    # test_case()

    print("Please set Dota2 resolution to 1920x1080 (16:9) in borderless windowed mode and run this program in administrator mode")
    print("Enter the Minesweeper mini-game interface, move the mouse over the 'Start' button and press 'b' to start")
    print()
    print("Hotkeys:")
    print("'b': Before entering each level, move the mouse over the 'Start/Continue' button and press 'b' to start")
    print("'f': Toggle auto-flagging of mines (default off)")
    print("'p': Pause/Resume the program")
    print("'esc': Exit the program")
    print("'t': Toggle turbo mode，Trade off between speed and accuracy. Turn on for higher score and off for program stability (default off)")

    listener = Listener(on_press=on_press)
    listener.start()

    try:
        run()
    except Exception as e:
        print('An error occurred:', e)
        time.sleep(3)
        raise
