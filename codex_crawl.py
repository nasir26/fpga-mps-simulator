"""
CODEX CRAWL: The Archive of Failure
====================================
An original 2D turn-based dungeon game built only with Pygame + the
Python standard library.

CORE IDEA
---------
Every hazard in this dungeon (magic runes, sentry turrets, hunting
guardians, collapsing floors) obeys a hidden rule that is never
explained up front. The only way to learn a rule is to break it and
die. Death is cheap (you respawn instantly at the level's start) but
never silent: the game always tells you *exactly* why you died, in
concrete terms, and permanently records that lesson in the Archive
(saved to codex_archive.json next to this script). Knowledge earned
through failure carries across the whole run - and across future
sessions, since the Archive persists on disk.

CONTROLS
--------
  Arrow keys / WASD  - move one tile (also attacks nothing, this is a
                        puzzle-dungeon, not combat)
  Space              - wait one turn in place (useful for timing)
  Tab                - open/close the full Archive of everything you
                        have ever learned
  Enter / Space      - continue past death / level-clear / win screens
  Esc                - quit

Run with:  python3 codex_crawl.py
"""

import json
import os
import random
import sys

import pygame

# --------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------

TILE = 38
PANEL_W = 340
TOP_BAR = 64
BOTTOM_BAR = 34

ARCHIVE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "codex_archive.json")

DIRS = {
    pygame.K_UP: (0, -1), pygame.K_w: (0, -1),
    pygame.K_DOWN: (0, 1), pygame.K_s: (0, 1),
    pygame.K_LEFT: (-1, 0), pygame.K_a: (-1, 0),
    pygame.K_RIGHT: (1, 0), pygame.K_d: (1, 0),
}

RUNE_COLORS = {
    "a": ("Crimson Rune", (214, 64, 64)),
    "b": ("Verdant Rune", (72, 176, 90)),
    "c": ("Azure Rune", (74, 130, 220)),
}

COL_BG = (17, 18, 24)
COL_WALL = (52, 54, 66)
COL_WALL_EDGE = (34, 35, 44)
COL_FLOOR = (34, 36, 46)
COL_FLOOR_ALT = (39, 41, 52)
COL_PANEL = (24, 25, 32)
COL_TEXT = (226, 228, 235)
COL_DIM = (140, 143, 156)
COL_GOOD = (120, 220, 150)
COL_BAD = (235, 90, 90)
COL_WARN = (240, 190, 90)
COL_PLAYER = (110, 220, 235)
COL_EXIT = (120, 220, 150)
COL_START = (90, 100, 120)
COL_SHARD = (240, 210, 90)
COL_TURRET = (90, 92, 108)
COL_BOLT = (250, 120, 90)
COL_GUARDIAN = (190, 110, 230)
COL_PIT_STABLE = (70, 55, 45)
COL_PIT_WEAK = (150, 90, 40)
COL_PIT_GONE = (5, 5, 8)


# --------------------------------------------------------------------------
# Hidden rules
# --------------------------------------------------------------------------

class Rule:
    """Base class for a hazard's hidden win/lose condition."""

    def describe(self):
        raise NotImplementedError

    def is_safe(self, shards):
        raise NotImplementedError


class EvenShards(Rule):
    def describe(self):
        return "safe only while your Shard count is EVEN"

    def is_safe(self, shards):
        return shards % 2 == 0


class OddShards(Rule):
    def describe(self):
        return "safe only while your Shard count is ODD"

    def is_safe(self, shards):
        return shards % 2 == 1


class MinShards(Rule):
    def __init__(self, n):
        self.n = n

    def describe(self):
        return "safe only once your Shard count is at least {}".format(self.n)

    def is_safe(self, shards):
        return shards >= self.n


class MaxShards(Rule):
    def __init__(self, n):
        self.n = n

    def describe(self):
        return "safe only while your Shard count is at most {}".format(self.n)

    def is_safe(self, shards):
        return shards <= self.n


RULE_POOL = [
    lambda rng: EvenShards(),
    lambda rng: OddShards(),
    lambda rng: MinShards(rng.randint(1, 3)),
    lambda rng: MaxShards(rng.randint(0, 2)),
]

GUARDIAN_RULES = {
    "mirror": "moves the exact OPPOSITE of your last move (you go up, it goes down)",
    "copy": "moves in the SAME direction as your last move, one beat behind",
    "chase": "always steps directly toward you along its longer axis, every turn",
    "delayed": "repeats whatever move you made TWO turns ago",
}


# --------------------------------------------------------------------------
# Archive: the persistent memory of failure
# --------------------------------------------------------------------------

class Archive:
    def __init__(self, path):
        self.path = path
        self.entries = {}
        self.stats = {"total_deaths": 0, "levels_cleared": 0}
        self.load()

    def load(self):
        if os.path.exists(self.path):
            try:
                with open(self.path, "r") as f:
                    data = json.load(f)
                self.entries = data.get("entries", {})
                self.stats = data.get("stats", self.stats)
            except (ValueError, OSError):
                self.entries = {}

    def save(self):
        try:
            with open(self.path, "w") as f:
                json.dump({"entries": self.entries, "stats": self.stats}, f, indent=2)
        except OSError:
            pass

    def is_discovered(self, hid):
        return self.entries.get(hid, {}).get("discovered", False)

    def record_death(self, hid, level_name, label, rule_text):
        entry = self.entries.setdefault(hid, {
            "label": label, "level": level_name, "rule": rule_text,
            "discovered": False, "deaths": 0,
        })
        is_new = not entry["discovered"]
        entry["discovered"] = True
        entry["rule"] = rule_text
        entry["deaths"] += 1
        self.stats["total_deaths"] += 1
        self.save()
        return is_new

    def register(self, hid, level_name, label):
        """Make sure a locked (undiscovered) entry exists so the Archive can list '???'."""
        self.entries.setdefault(hid, {
            "label": label, "level": level_name, "rule": "",
            "discovered": False, "deaths": 0,
        })


# --------------------------------------------------------------------------
# Level construction helpers (rectangles avoid hand-counted ASCII errors)
# --------------------------------------------------------------------------

def build_grid(width, height, rooms, corridors, extras):
    grid = [["#"] * width for _ in range(height)]

    def carve(x0, y0, x1, y1):
        for y in range(y0, y1 + 1):
            for x in range(x0, x1 + 1):
                grid[y][x] = "."

    for rect in rooms:
        carve(*rect)
    for rect in corridors:
        carve(*rect)
    for (x, y, ch) in extras:
        grid[y][x] = ch
    return ["".join(row) for row in grid]


class LevelBlueprint:
    def __init__(self, name, rows, pit_delay=None, turret_period=None):
        self.name = name
        self.rows = rows
        self.width = len(rows[0])
        self.height = len(rows)
        assert all(len(r) == self.width for r in rows), "row width mismatch in " + name
        self.pit_delay_override = pit_delay
        self.turret_period_override = turret_period


DIR_CHARS = {"u": (0, -1), "d": (0, 1), "l": (-1, 0), "r": (1, 0)}


def define_levels():
    levels = []

    # Level 0 - "First Lesson": runes only.
    rows = build_grid(
        19, 11,
        rooms=[(1, 3, 5, 7), (9, 1, 17, 9)],
        corridors=[(6, 5, 8, 5)],
        extras=[(2, 5, "S"), (3, 3, "$"), (12, 3, "a"), (14, 7, "b"), (16, 5, "E")],
    )
    levels.append(LevelBlueprint("First Lesson", rows))

    # Level 1 - "The Watcher": runes + a turret guarding a chokepoint.
    rows = build_grid(
        19, 11,
        rooms=[(1, 3, 4, 7), (13, 3, 17, 7)],
        corridors=[(5, 5, 12, 5)],
        extras=[(2, 5, "S"), (5, 4, "d"), (15, 5, "a"), (17, 5, "E")],
    )
    levels.append(LevelBlueprint("The Watcher", rows, turret_period=4))

    # Level 2 - "Shadow and Steel": guardian arena + turret chokepoint.
    rows = build_grid(
        21, 11,
        rooms=[(1, 3, 4, 7), (6, 1, 12, 9), (17, 3, 19, 7)],
        corridors=[(5, 5, 5, 5), (13, 5, 16, 5)],
        extras=[(2, 5, "S"), (9, 5, "g"), (10, 3, "$"), (10, 7, "b"),
                (13, 4, "d"), (18, 5, "E")],
    )
    levels.append(LevelBlueprint("Shadow and Steel", rows))

    # Level 3 - "The Crumbling Path": collapsing floor tiles + runes.
    rows = build_grid(
        19, 11,
        rooms=[(1, 3, 4, 7), (9, 1, 17, 9)],
        corridors=[(5, 5, 8, 5)],
        extras=[(2, 5, "S"), (3, 4, "$"), (6, 5, "p"), (7, 5, "p"),
                (12, 3, "a"), (14, 7, "c"), (16, 5, "E")],
    )
    levels.append(LevelBlueprint("The Crumbling Path", rows, pit_delay=3))

    # Level 4 - "The Final Trial": everything at once.
    rows = build_grid(
        23, 13,
        rooms=[(1, 5, 4, 9), (6, 1, 12, 11), (18, 4, 21, 10)],
        corridors=[(5, 7, 5, 7), (13, 6, 17, 6)],
        extras=[
            (2, 7, "S"),
            (9, 6, "g"),
            (8, 3, "a"),
            (10, 9, "$"),
            (13, 5, "d"),
            (15, 6, "p"), (16, 6, "p"),
            (19, 5, "b"), (20, 8, "c"),
            (20, 6, "E"),
        ],
    )
    levels.append(LevelBlueprint("The Final Trial", rows, pit_delay=3, turret_period=4))

    return levels


# --------------------------------------------------------------------------
# Runtime hazard objects
# --------------------------------------------------------------------------

class Turret:
    def __init__(self, hid, pos, direction, period):
        self.hid = hid
        self.pos = pos
        self.dir = direction
        self.period = period
        self.counter = 0
        self.bolt = None

    def label(self):
        return "Sentry Turret"

    def describe_rule(self):
        return "fires a bolt down its lane every {} turns (barrel flashes the turn before)".format(self.period)

    def charging(self):
        return self.counter == self.period - 1

    def reset(self):
        self.counter = 0
        self.bolt = None

    def tick(self, walls, width, height):
        hit = None
        if self.bolt is not None:
            nx, ny = self.bolt[0] + self.dir[0], self.bolt[1] + self.dir[1]
            if (nx, ny) in walls or not (0 <= nx < width and 0 <= ny < height):
                self.bolt = None
            else:
                self.bolt = (nx, ny)
                hit = self.bolt

        # Counter advances every turn regardless of an in-flight bolt, so the
        # firing cadence always matches the period stated in describe_rule().
        self.counter += 1
        if self.counter >= self.period:
            self.counter = 0
            if self.bolt is None:
                start = (self.pos[0] + self.dir[0], self.pos[1] + self.dir[1])
                if start not in walls:
                    self.bolt = start
                    hit = self.bolt
        return hit


class Guardian:
    def __init__(self, hid, spawn, rule_id):
        self.hid = hid
        self.spawn = spawn
        self.pos = spawn
        self.rule_id = rule_id
        self.history = []

    def label(self):
        return "Guardian"

    def describe_rule(self):
        return GUARDIAN_RULES[self.rule_id]

    def reset(self):
        self.pos = self.spawn
        self.history = []

    def note_player_move(self, move):
        self.history.append(move)
        if len(self.history) > 3:
            self.history.pop(0)

    def tick(self, walls, width, height):
        move = (0, 0)
        if self.rule_id == "mirror" and self.history:
            last = self.history[-1]
            move = (-last[0], -last[1])
        elif self.rule_id == "copy" and self.history:
            move = self.history[-1]
        elif self.rule_id == "delayed" and len(self.history) >= 2:
            move = self.history[-2]
        elif self.rule_id == "chase":
            move = (0, 0)  # placeholder, real chase computed by caller with player pos
        if move != (0, 0):
            nx, ny = self.pos[0] + move[0], self.pos[1] + move[1]
            if (nx, ny) not in walls and 0 <= nx < width and 0 <= ny < height:
                self.pos = (nx, ny)

    def chase_toward(self, player_pos, walls, width, height):
        dx = player_pos[0] - self.pos[0]
        dy = player_pos[1] - self.pos[1]
        candidates = []
        if abs(dx) >= abs(dy) and dx != 0:
            candidates.append((1 if dx > 0 else -1, 0))
        if dy != 0:
            candidates.append((0, 1 if dy > 0 else -1))
        if dx != 0:
            candidates.append((1 if dx > 0 else -1, 0))
        for mv in candidates:
            nx, ny = self.pos[0] + mv[0], self.pos[1] + mv[1]
            if (nx, ny) not in walls and 0 <= nx < width and 0 <= ny < height:
                self.pos = (nx, ny)
                return


class Pit:
    STABLE, WEAK, GONE = "stable", "weak", "gone"

    def __init__(self, hid, pos, delay):
        self.hid = hid
        self.pos = pos
        self.delay = delay
        self.state = Pit.STABLE
        self.timer = 0

    def label(self):
        return "Collapsing Floor"

    def describe_rule(self):
        return "collapses {} turns after first being stepped on - keep moving".format(self.delay)

    def reset(self):
        self.state = Pit.STABLE
        self.timer = 0

    def on_step(self):
        if self.state == Pit.STABLE:
            self.state = Pit.WEAK
            self.timer = self.delay

    def tick(self):
        if self.state == Pit.WEAK:
            self.timer -= 1
            if self.timer <= 0:
                self.state = Pit.GONE


# --------------------------------------------------------------------------
# Level: parsed static data + live hazard instances
# --------------------------------------------------------------------------

class Level:
    def __init__(self, blueprint, index):
        self.blueprint = blueprint
        self.index = index
        self.name = blueprint.name
        self.width = blueprint.width
        self.height = blueprint.height

        self.walls = set()
        self.floors = set()
        self.start = None
        self.exit = None
        self.rune_tiles = {}      # pos -> color key
        self.rune_colors_seen = []
        self.shard_tiles = set()
        self.turrets = []
        self.guardians = []
        self.pits = []

        rng = random.Random("codex-crawl-level-{}".format(index))

        turret_i = 0
        guardian_i = 0
        for y, row in enumerate(blueprint.rows):
            for x, ch in enumerate(row):
                pos = (x, y)
                if ch == "#":
                    self.walls.add(pos)
                elif ch == ".":
                    self.floors.add(pos)
                elif ch == "S":
                    self.start = pos
                    self.floors.add(pos)
                elif ch == "E":
                    self.exit = pos
                    self.floors.add(pos)
                elif ch in RUNE_COLORS:
                    self.rune_tiles[pos] = ch
                    if ch not in self.rune_colors_seen:
                        self.rune_colors_seen.append(ch)
                    self.floors.add(pos)
                elif ch == "$":
                    self.shard_tiles.add(pos)
                    self.floors.add(pos)
                elif ch in DIR_CHARS:
                    hid = "L{}_turret_{}".format(index, turret_i)
                    period = blueprint.turret_period_override or rng.randint(3, 5)
                    self.turrets.append(Turret(hid, pos, DIR_CHARS[ch], period))
                    turret_i += 1
                    self.walls.add(pos)
                elif ch == "g":
                    hid = "L{}_guardian_{}".format(index, guardian_i)
                    rule_id = rng.choice(list(GUARDIAN_RULES.keys()))
                    self.guardians.append(Guardian(hid, pos, rule_id))
                    guardian_i += 1
                    self.floors.add(pos)
                elif ch == "p":
                    delay = blueprint.pit_delay_override or rng.randint(2, 4)
                    self.pits.append(Pit("L{}_pit".format(index), pos, delay))
                    self.floors.add(pos)
                else:
                    raise ValueError("Unknown level tile {!r} at {}".format(ch, pos))

        assert self.start and self.exit, "level {} missing S or E".format(index)

        self.rune_rules = {}
        for c in self.rune_colors_seen:
            builder = rng.choice(RULE_POOL)
            self.rune_rules[c] = builder(rng)

        self.pit_hid = "L{}_pit".format(index)
        self.validate_reachable()

    def validate_reachable(self):
        seen = {self.start}
        stack = [self.start]
        passable = self.floors | set(self.rune_tiles.keys()) | self.shard_tiles
        while stack:
            cx, cy = stack.pop()
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                np_ = (cx + dx, cy + dy)
                if np_ in passable and np_ not in seen:
                    seen.add(np_)
                    stack.append(np_)
        required = {self.exit} | set(self.rune_tiles) | self.shard_tiles
        for g in self.guardians:
            required.add(g.spawn)
        for p in self.pits:
            required.add(p.pos)
        missing = required - seen
        if missing:
            raise AssertionError("level {} has unreachable tiles: {}".format(self.index, missing))

    def is_wall(self, pos):
        return pos in self.walls

    def reset_hazards(self):
        for t in self.turrets:
            t.reset()
        for g in self.guardians:
            g.reset()
        for p in self.pits:
            p.reset()

    def register_locked_entries(self, archive):
        for c in self.rune_colors_seen:
            label, _ = RUNE_COLORS[c]
            archive.register("L{}_rune_{}".format(self.index, c), self.name, label)
        for t in self.turrets:
            archive.register(t.hid, self.name, t.label())
        for g in self.guardians:
            archive.register(g.hid, self.name, g.label())
        if self.pits:
            archive.register(self.pit_hid, self.name, "Collapsing Floor")


# --------------------------------------------------------------------------
# Game state machine
# --------------------------------------------------------------------------

class Game:
    MENU, PLAYING, DEATH, LEVEL_CLEAR, ARCHIVE_VIEW, WIN = range(6)

    def __init__(self):
        self.blueprints = define_levels()
        self.archive = Archive(ARCHIVE_PATH)
        self.level_index = 0
        self.level = Level(self.blueprints[0], 0)
        self.player = self.level.start
        self.shards = 0
        self.deaths_this_level = 0
        self.last_move = (0, 0)
        self.message = "Find the exit. Every death teaches you something. Press any key."
        self.message_kind = "info"
        self.death_info = None
        self.state = Game.MENU
        self.pre_archive_state = Game.PLAYING
        self.max_w = max(bp.width for bp in self.blueprints)
        self.max_h = max(bp.height for bp in self.blueprints)
        for bp_i, bp in enumerate(self.blueprints):
            Level(bp, bp_i).register_locked_entries(self.archive)
        self.level.register_locked_entries(self.archive)

    # -- level lifecycle -------------------------------------------------

    def load_level(self, index):
        self.level_index = index
        self.level = Level(self.blueprints[index], index)
        self.level.register_locked_entries(self.archive)
        self.player = self.level.start
        self.shards = 0
        self.deaths_this_level = 0
        self.last_move = (0, 0)
        self.message = "{} - {}".format(index + 1, self.level.name)
        self.message_kind = "info"
        self.state = Game.PLAYING

    def respawn(self):
        self.level.reset_hazards()
        self.player = self.level.start
        self.shards = 0
        self.last_move = (0, 0)
        self.state = Game.PLAYING

    # -- turn resolution ---------------------------------------------------

    def try_move(self, dx, dy):
        if self.state != Game.PLAYING:
            return
        target = (self.player[0] + dx, self.player[1] + dy)
        self.take_turn(target, (dx, dy))

    def wait_turn(self):
        if self.state != Game.PLAYING:
            return
        self.take_turn(self.player, (0, 0))

    def take_turn(self, target, move_vec):
        level = self.level
        if move_vec != (0, 0) and level.is_wall(target):
            return  # bump into a wall: free, no turn consumed

        prev_player = self.player
        moved = (target != self.player)
        if moved:
            self.player = target
        self.last_move = move_vec

        # immediate tile effects for the player's own move
        died = None  # (hazard_kind, hid, label, rule_text, flavor)

        if self.player in level.shard_tiles:
            level.shard_tiles.discard(self.player)
            self.shards += 1
            self.message = "Shard collected. Shards: {}".format(self.shards)
            self.message_kind = "info"
        elif self.player in level.rune_tiles:
            color = level.rune_tiles[self.player]
            rule = level.rune_rules[color]
            label, _ = RUNE_COLORS[color]
            if not rule.is_safe(self.shards):
                died = ("rune", "L{}_rune_{}".format(level.index, color), label,
                        rule.describe(),
                        "You stepped on the {} with {} Shard(s) and it was not {}.".format(
                            label, self.shards, rule.describe()))
        else:
            for p in level.pits:
                if p.pos == self.player:
                    if p.state == Pit.GONE:
                        died = ("pit", p.hid, "Collapsing Floor", p.describe_rule(),
                                "You stepped into a floor that had already collapsed.")
                    else:
                        p.on_step()
                    break

        # world tick (only if the player is still alive and actually acted)
        if died is None:
            for t in level.turrets:
                hit = t.tick(level.walls, level.width, level.height)
                if hit == self.player:
                    died = ("turret", t.hid, t.label(), t.describe_rule(),
                            "A bolt from the {} caught you in its lane.".format(t.label()))

        if died is None:
            for g in level.guardians:
                if g.rule_id == "chase":
                    before = g.pos
                    g.chase_toward(self.player, level.walls, level.width, level.height)
                else:
                    g.note_player_move(move_vec)
                    before = g.pos
                    g.tick(level.walls, level.width, level.height)
                after = g.pos
                caught = (after == self.player) or (after == prev_player and before == self.player and moved)
                if caught:
                    died = ("guardian", g.hid, g.label(), g.describe_rule(),
                            "The Guardian caught you. It {}.".format(g.describe_rule()))

        if died is None:
            for p in level.pits:
                p.tick()
                if p.pos == self.player and p.state == Pit.GONE:
                    died = ("pit", p.hid, "Collapsing Floor", p.describe_rule(),
                            "The floor gave way beneath you.")

        if died:
            self.handle_death(*died)
            return

        if self.player == level.exit:
            self.handle_level_clear()

    def handle_death(self, kind, hid, label, rule_text, flavor):
        is_new = self.archive.record_death(hid, self.level.name, label, rule_text)
        self.deaths_this_level += 1
        self.death_info = {
            "label": label, "rule": rule_text, "flavor": flavor, "is_new": is_new,
        }
        self.state = Game.DEATH

    def handle_level_clear(self):
        self.archive.stats["levels_cleared"] = max(
            self.archive.stats.get("levels_cleared", 0), self.level_index + 1)
        self.archive.save()
        if self.level_index + 1 >= len(self.blueprints):
            self.state = Game.WIN
        else:
            self.state = Game.LEVEL_CLEAR

    def continue_after_death(self):
        self.respawn()

    def continue_after_clear(self):
        self.load_level(self.level_index + 1)

    def toggle_archive(self):
        if self.state == Game.ARCHIVE_VIEW:
            self.state = self.pre_archive_state
        else:
            self.pre_archive_state = self.state
            self.state = Game.ARCHIVE_VIEW


# --------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------

def draw_text(surface, font, text, pos, color=COL_TEXT):
    surface.blit(font.render(text, True, color), pos)


def wrap_text(text, font, max_width):
    words = text.split(" ")
    lines = []
    cur = ""
    for w in words:
        trial = (cur + " " + w).strip()
        if font.size(trial)[0] > max_width and cur:
            lines.append(cur)
            cur = w
        else:
            cur = trial
    if cur:
        lines.append(cur)
    return lines


class Renderer:
    def __init__(self, screen, font, font_small, font_big):
        self.screen = screen
        self.font = font
        self.font_small = font_small
        self.font_big = font_big

    def grid_to_px(self, pos):
        return TOP_BAR + pos[0] * TILE, TOP_BAR + pos[1] * TILE  # x offset uses TOP_BAR margin too

    def draw(self, game):
        self.screen.fill(COL_BG)
        if game.state == Game.MENU:
            self.draw_menu(game)
        elif game.state == Game.WIN:
            self.draw_win(game)
        else:
            self.draw_level(game)
            self.draw_panel(game)
            self.draw_top_bar(game)
            self.draw_bottom_bar(game)
            if game.state == Game.DEATH:
                self.draw_death_overlay(game)
            elif game.state == Game.LEVEL_CLEAR:
                self.draw_clear_overlay(game)
            elif game.state == Game.ARCHIVE_VIEW:
                self.draw_archive_overlay(game)

    def field_origin(self):
        return 20, TOP_BAR

    def draw_level(self, game):
        ox, oy = self.field_origin()
        level = game.level
        for y in range(level.height):
            for x in range(level.width):
                pos = (x, y)
                rect = pygame.Rect(ox + x * TILE, oy + y * TILE, TILE, TILE)
                if pos in level.walls:
                    pygame.draw.rect(self.screen, COL_WALL, rect)
                    pygame.draw.rect(self.screen, COL_WALL_EDGE, rect, 1)
                else:
                    base = COL_FLOOR_ALT if (x + y) % 2 == 0 else COL_FLOOR
                    pygame.draw.rect(self.screen, base, rect)

        if level.exit:
            self.tile_marker(ox, oy, level.exit, COL_EXIT, "E")
        self.tile_marker(ox, oy, level.start, COL_START, "S")

        for pos, color in level.rune_tiles.items():
            label, rgb = RUNE_COLORS[color]
            hid = "L{}_rune_{}".format(level.index, color)
            known = game.archive.is_discovered(hid)
            cx = ox + pos[0] * TILE + TILE // 2
            cy = oy + pos[1] * TILE + TILE // 2
            pygame.draw.circle(self.screen, rgb, (cx, cy), TILE // 2 - 5)
            mark = "!" if known else "?"
            draw_text(self.screen, self.font_small, mark,
                      (cx - 4, cy - 9), (10, 10, 10))

        for pos in level.shard_tiles:
            cx = ox + pos[0] * TILE + TILE // 2
            cy = oy + pos[1] * TILE + TILE // 2
            pts = [(cx, cy - 10), (cx + 8, cy), (cx, cy + 10), (cx - 8, cy)]
            pygame.draw.polygon(self.screen, COL_SHARD, pts)

        for p in level.pits:
            rect = pygame.Rect(ox + p.pos[0] * TILE, oy + p.pos[1] * TILE, TILE, TILE)
            if p.state == Pit.STABLE:
                pygame.draw.rect(self.screen, COL_PIT_STABLE, rect)
            elif p.state == Pit.WEAK:
                pygame.draw.rect(self.screen, COL_PIT_WEAK, rect)
            else:
                pygame.draw.ellipse(self.screen, COL_PIT_GONE, rect.inflate(-4, -4))

        for t in level.turrets:
            rect = pygame.Rect(ox + t.pos[0] * TILE, oy + t.pos[1] * TILE, TILE, TILE)
            color = COL_WARN if t.charging() else COL_TURRET
            pygame.draw.rect(self.screen, color, rect)
            cx, cy = rect.center
            bx, by = cx + t.dir[0] * TILE // 3, cy + t.dir[1] * TILE // 3
            pygame.draw.line(self.screen, (20, 20, 20), (cx, cy), (bx, by), 4)
            if t.bolt:
                bcx = ox + t.bolt[0] * TILE + TILE // 2
                bcy = oy + t.bolt[1] * TILE + TILE // 2
                pygame.draw.circle(self.screen, COL_BOLT, (bcx, bcy), TILE // 4)

        for g in level.guardians:
            cx = ox + g.pos[0] * TILE + TILE // 2
            cy = oy + g.pos[1] * TILE + TILE // 2
            pts = [(cx, cy - 12), (cx + 11, cy + 9), (cx - 11, cy + 9)]
            pygame.draw.polygon(self.screen, COL_GUARDIAN, pts)

        px = ox + game.player[0] * TILE + TILE // 2
        py = oy + game.player[1] * TILE + TILE // 2
        pygame.draw.circle(self.screen, COL_PLAYER, (px, py), TILE // 2 - 4)

    def tile_marker(self, ox, oy, pos, color, letter):
        rect = pygame.Rect(ox + pos[0] * TILE, oy + pos[1] * TILE, TILE, TILE)
        pygame.draw.rect(self.screen, color, rect, 3)

    def field_pixel_size(self):
        return self.field_origin()[0] * 2 + self.max_w_cached * TILE

    def draw_panel(self, game):
        w, h = self.screen.get_size()
        panel_x = w - PANEL_W
        pygame.draw.rect(self.screen, COL_PANEL, (panel_x, 0, PANEL_W, h))
        x = panel_x + 20
        y = 20
        draw_text(self.screen, self.font_big, "Level {}".format(game.level_index + 1), (x, y))
        y += 34
        draw_text(self.screen, self.font, game.level.name, (x, y), COL_DIM)
        y += 34
        draw_text(self.screen, self.font, "Shards: {}".format(game.shards), (x, y), COL_SHARD)
        y += 26
        draw_text(self.screen, self.font, "Deaths this level: {}".format(game.deaths_this_level), (x, y), COL_DIM)
        y += 26
        draw_text(self.screen, self.font, "Total deaths (ever): {}".format(game.archive.stats["total_deaths"]),
                   (x, y), COL_DIM)
        y += 40
        draw_text(self.screen, self.font, "Hazards in this level:", (x, y), COL_TEXT)
        y += 28
        hids = []
        for c in game.level.rune_colors_seen:
            label, rgb = RUNE_COLORS[c]
            hids.append(("L{}_rune_{}".format(game.level.index, c), label, rgb))
        for t in game.level.turrets:
            hids.append((t.hid, t.label(), COL_TURRET))
        for g in game.level.guardians:
            hids.append((g.hid, g.label(), COL_GUARDIAN))
        if game.level.pits:
            hids.append((game.level.pit_hid, "Collapsing Floor", COL_PIT_WEAK))

        for hid, label, rgb in hids:
            entry = game.archive.entries.get(hid, {})
            known = entry.get("discovered", False)
            pygame.draw.circle(self.screen, rgb, (x + 6, y + 8), 6)
            tag = label if known else label + "  (???)"
            draw_text(self.screen, self.font_small, tag, (x + 20, y), COL_GOOD if known else COL_DIM)
            y += 20
            if known:
                for line in wrap_text(entry.get("rule", ""), self.font_small, PANEL_W - 60):
                    draw_text(self.screen, self.font_small, "  " + line, (x + 20, y), COL_DIM)
                    y += 17
            y += 6

        y = h - 110
        draw_text(self.screen, self.font_small, "Arrows/WASD move, Space wait", (x, y), COL_DIM)
        y += 18
        draw_text(self.screen, self.font_small, "Tab: full Archive   Esc: quit", (x, y), COL_DIM)

    def draw_top_bar(self, game):
        w, _ = self.screen.get_size()
        pygame.draw.rect(self.screen, COL_PANEL, (0, 0, w - PANEL_W, TOP_BAR - 8))
        color = {"info": COL_TEXT, "warn": COL_WARN, "bad": COL_BAD, "good": COL_GOOD}.get(
            game.message_kind, COL_TEXT)
        draw_text(self.screen, self.font, game.message, (20, 18), color)

    def draw_bottom_bar(self, game):
        w, h = self.screen.get_size()
        pygame.draw.rect(self.screen, COL_PANEL, (0, h - BOTTOM_BAR, w - PANEL_W, BOTTOM_BAR))
        draw_text(self.screen, self.font_small,
                  "CODEX CRAWL - failure is never wasted", (20, h - BOTTOM_BAR + 8), COL_DIM)

    def overlay_box(self, title, title_color):
        w, h = self.screen.get_size()
        box = pygame.Rect(w // 2 - 320, h // 2 - 180, 640, 360)
        pygame.draw.rect(self.screen, (0, 0, 0), (0, 0, w, h), 0)
        s = pygame.Surface((w, h), pygame.SRCALPHA)
        s.fill((0, 0, 0, 180))
        self.screen.blit(s, (0, 0))
        pygame.draw.rect(self.screen, COL_PANEL, box)
        pygame.draw.rect(self.screen, title_color, box, 3)
        draw_text(self.screen, self.font_big, title, (box.x + 24, box.y + 20), title_color)
        return box

    def draw_death_overlay(self, game):
        box = self.overlay_box("YOU FAILED", COL_BAD)
        info = game.death_info
        y = box.y + 70
        tag = "NEW LESSON LEARNED" if info["is_new"] else "REMINDER"
        draw_text(self.screen, self.font, tag, (box.x + 24, y), COL_WARN if info["is_new"] else COL_DIM)
        y += 32
        for line in wrap_text(info["flavor"], self.font, box.w - 48):
            draw_text(self.screen, self.font, line, (box.x + 24, y), COL_TEXT)
            y += 24
        y += 10
        draw_text(self.screen, self.font, "{}: {}".format(info["label"], info["rule"]), (box.x + 24, y), COL_GOOD)
        y = box.y + box.h - 40
        draw_text(self.screen, self.font, "Press Space / Enter to try again", (box.x + 24, y), COL_DIM)

    def draw_clear_overlay(self, game):
        box = self.overlay_box("LEVEL CLEARED", COL_GOOD)
        y = box.y + 70
        draw_text(self.screen, self.font, "You reached the exit of \"{}\".".format(game.level.name),
                   (box.x + 24, y), COL_TEXT)
        y += 30
        draw_text(self.screen, self.font, "Deaths this level: {}".format(game.deaths_this_level),
                   (box.x + 24, y), COL_DIM)
        y = box.y + box.h - 40
        draw_text(self.screen, self.font, "Press Space / Enter to continue", (box.x + 24, y), COL_DIM)

    def draw_archive_overlay(self, game):
        w, h = self.screen.get_size()
        box = pygame.Rect(60, 40, w - 120, h - 80)
        s = pygame.Surface((w, h), pygame.SRCALPHA)
        s.fill((0, 0, 0, 200))
        self.screen.blit(s, (0, 0))
        pygame.draw.rect(self.screen, COL_PANEL, box)
        pygame.draw.rect(self.screen, COL_WARN, box, 3)
        draw_text(self.screen, self.font_big, "THE ARCHIVE OF FAILURE", (box.x + 24, box.y + 16), COL_WARN)
        draw_text(self.screen, self.font_small, "Tab to close", (box.x + box.w - 120, box.y + 24), COL_DIM)

        y = box.y + 60
        by_level = {}
        for hid, e in game.archive.entries.items():
            by_level.setdefault(e["level"], []).append((hid, e))
        col_x = box.x + 24
        for bp in game.blueprints:
            entries = by_level.get(bp.name, [])
            if not entries:
                continue
            draw_text(self.screen, self.font, bp.name, (col_x, y), COL_TEXT)
            y += 24
            for hid, e in sorted(entries):
                if e["discovered"]:
                    line = "  [{}] {} - {} (died {}x)".format("known", e["label"], e["rule"], e["deaths"])
                    color = COL_GOOD
                else:
                    line = "  [ ? ] {} - not yet discovered".format(e["label"])
                    color = COL_DIM
                for wl in wrap_text(line, self.font_small, box.w - 60):
                    draw_text(self.screen, self.font_small, wl, (col_x, y), color)
                    y += 18
            y += 10
            if y > box.y + box.h - 40:
                break

    def draw_menu(self, game):
        w, h = self.screen.get_size()
        draw_text(self.screen, self.font_big, "CODEX CRAWL", (w // 2 - 140, h // 2 - 140), COL_WARN)
        draw_text(self.screen, self.font, "The Archive of Failure", (w // 2 - 130, h // 2 - 100), COL_TEXT)
        lines = [
            "Every trap, turret, guardian and collapsing floor in this dungeon",
            "follows a hidden rule. You will not be told the rule in advance.",
            "",
            "When you die, the game tells you exactly what killed you and why -",
            "and remembers it forever in your Archive, even between play sessions.",
            "",
            "Arrows/WASD to move, Space to wait, Tab for your Archive, Esc to quit.",
            "",
            "Press any key to begin.",
        ]
        y = h // 2 - 50
        for line in lines:
            draw_text(self.screen, self.font_small, line, (w // 2 - 260, y), COL_DIM)
            y += 22

    def draw_win(self, game):
        w, h = self.screen.get_size()
        draw_text(self.screen, self.font_big, "YOU ESCAPED THE ARCHIVE", (w // 2 - 240, h // 2 - 120), COL_GOOD)
        draw_text(self.screen, self.font,
                  "Total deaths along the way: {}".format(game.archive.stats["total_deaths"]),
                  (w // 2 - 160, h // 2 - 60), COL_TEXT)
        draw_text(self.screen, self.font,
                  "Every one of them taught you something that got you here.",
                  (w // 2 - 260, h // 2 - 30), COL_DIM)
        draw_text(self.screen, self.font_small, "Press Esc to quit, or Tab to browse your full Archive.",
                  (w // 2 - 220, h // 2 + 20), COL_DIM)


# --------------------------------------------------------------------------
# Main loop
# --------------------------------------------------------------------------

def main():
    pygame.init()
    pygame.display.set_caption("Codex Crawl: The Archive of Failure")

    game = Game()
    field_w = game.max_w * TILE + 40
    field_h = game.max_h * TILE
    width = field_w + PANEL_W
    height = TOP_BAR + field_h + BOTTOM_BAR
    screen = pygame.display.set_mode((width, height))

    font = pygame.font.SysFont(None, 22)
    font_small = pygame.font.SysFont(None, 18)
    font_big = pygame.font.SysFont(None, 32)
    renderer = Renderer(screen, font, font_small, font_big)
    renderer.max_w_cached = game.max_w

    clock = pygame.time.Clock()
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_TAB:
                    game.toggle_archive()
                elif game.state == Game.MENU:
                    game.state = Game.PLAYING
                elif game.state == Game.ARCHIVE_VIEW:
                    pass
                elif game.state == Game.DEATH:
                    if event.key in (pygame.K_SPACE, pygame.K_RETURN):
                        game.continue_after_death()
                elif game.state == Game.LEVEL_CLEAR:
                    if event.key in (pygame.K_SPACE, pygame.K_RETURN):
                        game.continue_after_clear()
                elif game.state == Game.PLAYING:
                    if event.key in DIRS:
                        dx, dy = DIRS[event.key]
                        game.try_move(dx, dy)
                    elif event.key == pygame.K_SPACE:
                        game.wait_turn()

        renderer.draw(game)
        pygame.display.flip()
        clock.tick(60)

    pygame.quit()
    sys.exit(0)


if __name__ == "__main__":
    main()
