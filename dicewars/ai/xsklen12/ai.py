from collections import deque
from copy import deepcopy
from logging import getLogger
from typing import List, Union, Tuple, Deque, Dict

from dicewars.client.ai_driver import BattleCommand, EndTurnCommand, TransferCommand
from dicewars.client.game.board import Board
from dicewars.ai.utils import possible_attacks, probability_of_successful_attack, probability_of_holding_area



class AI:
    # Maximum number of dice on a single field
    MAX_DICE = 8

    # Limit a number of turns for the Max^n algorithm
    MAX_N_TURNS_LIMIT = 5

    # Limit the total number of turns for the Max^n algorithm to ensure termination
    MAX_N_TOTAL_TURNS_LIMIT = 1000

    # Time threshold (in seconds) for the Max^n algorithm
    MAX_N_TIME_THRESHOLD = 1.0

    # Depth of the Max^n algorithm
    MAX_N_DEPTH = 5

    # Number of iterations of the Max^n algorithm
    MAX_N_ITERATIONS = 1

    # Weight of a largest region for the heuristic function
    LARGEST_REGION_HEURISTIC_WEIGHT = 50

    # Weight of a region for the heuristic function
    REGION_HEURISTIC_WEIGHT = 5

    # Attack probability threshold
    ATTACK_PROBABILITY_THRESHOLD = 0.2

    # Weight for a preference of attack from a largest region
    ATTACK_FROM_LARGEST_REGION_WEIGHT = 3

    # Limit for a number of players to use '..._2' parameters
    ATTACK_PLAYERS_LIMIT_2 = 2

    # Attack probability threshold for more players
    ATTACK_PROBABILITY_THRESHOLD_2 = 0.4

    # Weight for a preference of attack from a largest region for more players
    ATTACK_FROM_LARGEST_REGION_WEIGHT_2 = 2

    TRANSFER_PROBABILITY_THRESHOLD = 0.3

    BATTLE_COMMAND = 1

    TRANSFER_COMMAND = 2



    def __init__(self, player_name: int, board: Board, players_order: List[int], max_transfers: int) -> None:
        super().__init__()
        self.player_name = player_name
        self.board = board
        self.players_order = players_order
        self.logger = getLogger('AI xsklen12')
        self.max_transfers = max_transfers
        self.transfer_count = 0
        self.transfer_count_this_turn = 0



    def ai_turn(self, board: Board, nb_moves_this_turn: int, nb_transfers_this_turn: int, nb_turns_this_game: int, time_left: float)\
            -> Union[BattleCommand, EndTurnCommand, TransferCommand]:
        self.board = board
        self.transfer_count_this_turn = nb_transfers_this_turn

        if time_left >= self.MAX_N_TIME_THRESHOLD and nb_moves_this_turn < self.MAX_N_TURNS_LIMIT and nb_turns_this_game < self.MAX_N_TOTAL_TURNS_LIMIT:
            players = deque(self.players_order)
            players.reverse()
            while players[-1] != self.player_name:
                players.rotate(1)

            players_copy = deepcopy(players)
            for max_n_iteration in range(self.MAX_N_ITERATIONS - 1):
                players.extend(players_copy)

            turn, players_heuristics = self.__max_n(deepcopy(board), players, self.MAX_N_DEPTH)
            if turn is None:
                return EndTurnCommand()

            source, target, command_type = turn
            self.logger.debug(f'Max_n turn: {source} -> {target}.')
            return self.__perform_command(source, target, command_type, nb_transfers_this_turn)

        turns = self.__get_possible_turns(self.player_name, board)
        if turns:
            source, target, command_type = turns[0]
            self.logger.debug(f'Possible turn: {source} -> {target}.')
            return self.__perform_command(source, target, command_type, nb_transfers_this_turn)

        self.logger.debug('No more suitable turns.')
        return EndTurnCommand()



    def __perform_command(self, source: int, target: int, command_type: int, nb_transfers_this_turn: int) -> Union[BattleCommand, EndTurnCommand, TransferCommand]:
        if command_type == self.BATTLE_COMMAND:
            return BattleCommand(source, target)
        elif command_type == self.TRANSFER_COMMAND and nb_transfers_this_turn < self.max_transfers:
            return TransferCommand(source, target)
        else:
            return EndTurnCommand()



    def __get_possible_turns(self, player_name: int, board: Board) -> List[Tuple[int, int, int]]:
        if board.nb_players_alive() > self.ATTACK_PLAYERS_LIMIT_2:
            attack_probability_threshold = self.ATTACK_PROBABILITY_THRESHOLD_2
            attack_from_largest_region_weight = self.ATTACK_FROM_LARGEST_REGION_WEIGHT_2
        else:
            attack_probability_threshold = self.ATTACK_PROBABILITY_THRESHOLD
            attack_from_largest_region_weight = self.ATTACK_FROM_LARGEST_REGION_WEIGHT

        turns = []
        largest_region = self.__get_largest_region_for_player(player_name, board)

        for source, target in possible_attacks(board, player_name):
            source_name = source.get_name()
            target_name = target.get_name()
            attack_power = target.get_dice()

            turn_probability = probability_of_successful_attack(board, source_name, target_name)
            turn_probability *= probability_of_holding_area(board, target_name, attack_power, player_name)
            if turn_probability >= attack_probability_threshold or attack_power == self.MAX_DICE:
                if source_name in largest_region:
                    turn_probability *= attack_from_largest_region_weight
                turns.append((source_name, target_name, self.BATTLE_COMMAND, turn_probability))

        self.transfer_count = self.transfer_count_this_turn if player_name == self.player_name else 0
        border_area_names = [area.get_name() for area in board.get_player_border(player_name)]
        player_area_names = [area.get_name() for area in board.get_player_areas(player_name)]
        border_neighbor_area_names = [
            area.get_name() for area in board.get_player_areas(player_name)
            if area.get_name() not in border_area_names and list(filter(lambda neighbor_name: neighbor_name in border_area_names, area.get_adjacent_areas_names()))
        ]
        inner_area_names = [
            area.get_name() for area in board.get_player_areas(player_name)
            if area.get_name() not in border_area_names and area.get_name() not in border_neighbor_area_names]

        for area in board.get_player_areas(player_name):
            if self.transfer_count == self.max_transfers:
                break

            if area.get_name() in border_neighbor_area_names:
                for neighbor_area_name in area.get_adjacent_areas_names():
                    if neighbor_area_name in inner_area_names:
                        self.__request_transfer_from_area(
                            board,
                            neighbor_area_name,
                            area.get_name(),
                            area.get_name(),
                            inner_area_names,
                            border_area_names,
                            player_name,
                            [],
                            turns
                        )
            elif area.get_name() in border_area_names and area.get_dice() < self.MAX_DICE:
                for neighbor_area_name in area.get_adjacent_areas_names():
                    if neighbor_area_name not in player_area_names:
                        continue

                    neighbor_area = board.get_area(neighbor_area_name)
                    turn_probability = self.__get_transfer_probability(board, neighbor_area_name, area.get_name(), player_name)
                    if neighbor_area.get_dice() > 1 and turn_probability > self.TRANSFER_PROBABILITY_THRESHOLD and neighbor_area_name not in border_area_names:
                        turns.append((neighbor_area_name, area.get_name(), self.TRANSFER_COMMAND, turn_probability))
                        self.transfer_count += 1

        turns = sorted(turns, key = lambda turn: turn[3], reverse = True)
        turns = list(map(lambda turn: turn[:3], turns))
        return turns



    def __request_transfer_from_area(
        self,
        board: Board,
        source: int,
        target: int,
        original_target: int,
        inner_area_names: List[int],
        border_area_names: List[int],
        player_name: int,
        ignored_areas: List[int],
        turns: List[Tuple[int, int, int, float]]
    ) -> None:
        source_area = board.get_area(source)
        target_area = board.get_area(target)
        original_target_area = board.get_area(original_target)
        ignored_areas.append(source)

        for neighbor_area_name in source_area.get_adjacent_areas_names():
            if neighbor_area_name not in ignored_areas and neighbor_area_name in inner_area_names:
                self.__request_transfer_from_area(board, neighbor_area_name, source, original_target, inner_area_names, border_area_names, player_name, ignored_areas, turns)

        turn_probabilities = []
        for neighbor_area_name in original_target_area.get_adjacent_areas_names():
            if neighbor_area_name in border_area_names:
                turn_probabilities.append(self.__get_transfer_probability(board, source, neighbor_area_name, player_name))

        turn_probability = max(turn_probabilities)
        if self.transfer_count < self.max_transfers\
            and turn_probability > self.TRANSFER_PROBABILITY_THRESHOLD\
            and target_area.get_dice() < self.MAX_DICE\
            and source_area.get_dice() > 1:
            turns.append((source, target, self.TRANSFER_COMMAND, turn_probability))
            self.transfer_count += 1



    def __get_transfer_probability(self, board: Board, source: int, target: int, player_name: int) -> float:
        source_area = board.get_area(source)
        target_area = board.get_area(target)
        final_dice = min(target_area.get_dice() + source_area.get_dice(), self.MAX_DICE)
        original_dice = target_area.get_dice()

        enemy_area_names = [
            enemy_area_name for enemy_area_name in target_area.get_adjacent_areas_names()
            if board.get_area(enemy_area_name).get_owner_name() != player_name
        ]
        attack_probabilities = [1]
        target_area.set_dice(final_dice)
        for enemy_area_name in enemy_area_names:
            attack_probabilities.append(probability_of_successful_attack(board, target, enemy_area_name))
        target_area.set_dice(original_dice)

        return probability_of_holding_area(board, target_area.get_name(), final_dice, player_name) * max(attack_probabilities)



    @staticmethod
    def __get_largest_region_for_player(player_name: int, board: Board) -> List[int]:
        players_regions = board.get_players_regions(player_name)
        max_region_size = max(len(region) for region in players_regions)
        return list(filter(lambda region: len(region) == max_region_size, players_regions))[0]



    def __max_n(self, board: Board, players_names: Deque[int], depth: int) -> Tuple[Union[Tuple[int, int, int], None], Dict[int, float]]:
        if not players_names:
            return None, self.__get_players_heuristics([self.player_name], board)

        player_name = players_names[-1]
        if not board.get_player_areas(player_name):
            players_names.pop()
            return self.__max_n(deepcopy(board), players_names, depth)

        if depth:
            turns = self.__get_possible_turns(player_name, board)[:depth]
            if not turns:
                ignored_turn, players_heuristics = self.__max_n(deepcopy(board), players_names, 0)
                return None, players_heuristics

            players_heuristics = {}
            turn = 0, 0, self.BATTLE_COMMAND
            for source, target, command_type in turns:
                board_copy = deepcopy(board)
                self.__simulate_command(board_copy, source, target, command_type)
                ignored_turn, players_new_heuristics = self.__max_n(board_copy, players_names, depth - 1)

                if player_name in players_new_heuristics:
                    if player_name not in players_heuristics or players_new_heuristics[player_name] > players_heuristics[player_name]:
                        players_heuristics = deepcopy(players_new_heuristics)
                        turn = source, target, command_type

            return turn if players_heuristics else None, players_heuristics

        players_names.pop()
        if players_names:
            return self.__max_n(deepcopy(board), players_names, self.MAX_N_DEPTH)

        living_players = set(area.get_owner_name() for area in board.areas.values())
        players_heuristics = self.__get_players_heuristics(list(living_players), board)
        return None, players_heuristics



    def __simulate_command(self, board: Board, source_name: int, target_name: int, command_type: int) -> None:
        source = board.get_area(source_name)
        target = board.get_area(target_name)

        if command_type == self.BATTLE_COMMAND:
            target.set_dice(source.get_dice() - 1)
            source.set_dice(1)
            target.set_owner(source_name)
        elif command_type == self.TRANSFER_COMMAND:
            former_target_dice = target.get_dice()
            target.set_dice(min(target.get_dice() + source.get_dice(), self.MAX_DICE))
            source.set_dice(max(source.get_dice() - (target.get_dice() - former_target_dice), 1))



    def __get_players_heuristics(self, players: List[int], board: Board) -> Dict[int, float]:
        players_heuristics = {}
        for player in players:
            players_heuristics[player] = board.get_player_dice(player)
            player_regions = board.get_players_regions(player)
            player_regions_sizes = []

            for region in player_regions:
                region_size = len(region)
                players_heuristics[player] += self.REGION_HEURISTIC_WEIGHT * region_size
                player_regions_sizes.append(region_size)

            players_heuristics[player] += self.LARGEST_REGION_HEURISTIC_WEIGHT * max(player_regions_sizes)

        return players_heuristics