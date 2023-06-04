
from baselines.acer.flags import AcerFlags
from baselines.ppo2.flags import PPOFlags

class RogueAcerFlags(AcerFlags):

    CFG_sections = {'RNG', 'Training', 'Log', 'Rogue'}

    def __init__(self):
        super().__init__()

        self.policy = "Towers_LSTM"

        # It's important that nstack = 1
        self.nstack = 1

        self.nsteps = 60

        # Maximum number of steps in an episode
        self.max_episode_len = 500
        # Number of latest episodes to use for stats
        self.episodes_for_evaluation = 200

        # State generator name
        self.state_generator = "FullMap_5L_StateGenerator"
        # Reward generator name
        self.reward_generator = "StairsOnly_RewardGenerator"
        # Actions available
        self.actions = 'hjkl>'

        # Wheter to enable monsters
        self.use_monsters = False
        # Whether to enable hidden tiles
        self.enable_secrets = False
        # Level where the amulet of Yendor will be
        self.amulet_level = 26
        # Number of steps after which the rouge becomes faint
        self.hungertime = 1300
        # Maximum number of traps
        self.max_traps = 0
        # Whether to turn the descent action into ascent when the amulet level is reached
        self.transform_descent_action = True
        # Whether to send the refresh command after each action
        self.refresh_after_commands = False
        # Whether the dark rooms will be present in the game
        self.disable_dark_rooms = False
        # Whether the mazes will be present in the game
        self.disable_mazes = False
        # Whether to insert many more mazes in every level
        self.more_mazes = False
        # Which level the game will start at
        self.start_level = 1
class RoguePPOFlags(PPOFlags):

    CFG_sections = {'RNG', 'Training', 'Log', 'Rogue'}

    def __init__(self):
        super().__init__()

        self.policy = "Towers_LSTM"

        # It's important that nstack = 1
        self.nstack = 1

        self.nsteps = 60

        # Maximum number of steps in an episode
        self.max_episode_len = 500
        # Number of latest episodes to use for stats
        self.episodes_for_evaluation = 200

        # State generator name
        self.state_generator = "FullMap_5L_StateGenerator"
        # Reward generator name
        self.reward_generator = "StairsOnly_RewardGenerator"
        # Actions available
        self.actions = 'hjkl>'

        # Wheter to enable monsters
        self.use_monsters = False
        # Whether to enable hidden tiles
        self.enable_secrets = False
        # Level where the amulet of Yendor will be
        self.amulet_level = 26
        # Number of steps after which the rouge becomes faint
        self.hungertime = 1300
        # Maximum number of traps
        self.max_traps = 0
        # Whether to turn the descent action into ascent when the amulet level is reached
        self.transform_descent_action = True
        # Whether to send the refresh command after each action
        self.refresh_after_commands = False
        # Whether the dark rooms will be present in the game
        self.disable_dark_rooms = False
        # Whether the mazes will be present in the game
        self.disable_mazes = False
        # Whether to insert many more mazes in every level
        self.more_mazes = False
        # Which level the game will start at
        self.start_level = 1
