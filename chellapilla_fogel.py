from types import new_class
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import List, Tuple, Optional
from loguru import logger
import random

class CheckersBoard:
    def __init__(self):
        self.board = np.zeros(32, dtype=float)
        self.current_player = 1  # 1 for red, -1 for white
        self.initialize_board()
        
        self.board_mapping = {
            (i, j): pos for pos, (i, j) in enumerate([
                (x, y) for x in range(8) for y in range(8)
                if (x + y) % 2 == 1
            ])
        }

        self.reverse_mapping = {v:k for k, v in self.board_mapping.items()}
    
    def initialize_board(self):
        for i in range(12):
            self.board[i] = 1  # red
        for i in range(20, 32):
            self.board[i] = -1  # white
        
    def get_position(self, row: int, col: int) -> Optional[int]:
        return self.board_mapping.get((row, col))
    
    def is_valid_position(self, row: int, col: int) -> Optional[int]:
        return (row, col) in self.board_mapping
    
    def copy(self):
        new_board = CheckersBoard()
        new_board.board = self.board.copy()
        new_board.current_player = self.current_player
        return new_board
    
    def make_move(self, move: List[int]) -> None:
        """This makes a move."""
        start, end = move[0], move[-1]

        # Make the move
        self.board[end] = self.board[start]
        self.board[start] = 0

        # Handle jumps
        if abs(start - end) > 4:
            for i in range(len(move) - 1):
                jumped_pos = self.get_jumped_position(move[i], move[i + 1])
                self.board[jumped_pos] = 0
            
            # king promo
            if self.should_promote(end):
                self.board[end] *= 2
            
            # ... and switch players
            self.current_player *= -1
    
    def should_prompte(self, position: int) -> bool:
        """Check if a piece should be prompted to king"""
        row = self.reverse_mapping[position][0]
        piece = self.board[position]
        return (piece == 1 and row == 7) or (piece == -1 and row == 0)
    
    def get_jumped_position(self, start: int, end: int) -> int:
        """Get the position of the jumped piece"""
        start_row, start_col = self.reverse_mapping[start]
        end_row, end_col = self.reverse_mapping[end]
        jumped_row = (start_row + end_row) // 2
        jumped_col = (start_col + end_col) // 2
        return self.board_mapping[(jumped_row, jumped_col)]


class MoveGenerator:
    def __init__(self):
        self.directions = {
            1: [(-1, -1), (-1, 1)],  # red moves moving up
            -1: [(1, -1), (1, 1)],  # white moves (moving down)
            2: [(-1, -1), (-1, 1), (1, -1), (1, 1)],  # king moves
            -2: [(-1, -1), (-1, 1), (1, -1), (1, 1)]  # king moves
        }
    
    def get_legal_moves(self, board: CheckersBoard) -> List[List[int]]:
        """Get all legal moves for current players."""
        jumps = self.get_all_jumps(board)
        if jumps:
            return jumps
        
        # if no jumps available, get regular moves
        regular_moves = self.get_all_regular_moves(board)
        return regular_moves
    
    def get_all_jumps(self, board: CheckersBoard) -> List[List[int]]:
        """Get all possible jump moves"""
        all_jumps = []
        for pos in range(32):
            if board.board[pos] * board.current_player > 0:
                jumps = self.get_piece_jumps(board, pos, [pos])
                all_jumps.extend(jumps)
        return all_jumps
    
    def get_piece_jumps(self, board: CheckersBoard, pos: int,
    current_sequence: List[int]) -> List[List[int]]:
        """Get all possible jumps for a piece recursively"""
        jumps = []
        piece = board.board[pos]
        row, col = board.reverse_mapping[pos]

        for dr, dc in self.directions[piece]:
            jump_row, jump_col = row + 2*dr, col + 2*dc
            if not board.is_valid_position(jump_row, jump_col):
                continue

            middle_pos = board.get_position(row + dr, col + dc)
            jump_pos = board.get_position(jump_row, jump_col)

            if (middle_pos is not None and jump_pos is not None and board.board[middle_pos] * board.current_player < 0 and board.board[jump_pos] == 0):
                temp_board = boad.copy()
                temp_board.board[jump_pos] = temp_board.board[pos]
                temp_board.board[pos] = 0
                temp_board.board[middle_pos] = 0

                # check if there are additional jumps
                new_sequence = current_sequence + [jump_pos]
                additional_jumps = self.get_piece_jumps(temp_board, jump_pos, new_sequence)

                if additional_jumps:
                    jumps.extend(additional_jumps)
                else:
                    jumps.append(new_sequence)
        
        return jumps
    
    def get_all_regular_moves(self, board: CheckersBoard) -> List[List[int]]:
        """Get all psosible regular moves"""
        moves = []
        for pos in range(32):
            if board.board[pos] * board.current_player > 0:
                moves.extend(self.get_piece_regular_moves(board, pos))
        return moves

    def get_piece_regular_moves(self, board: CheckersBoard, pos: int) -> List[List[int]]:
        """Get all possible regular moves for a piece"""
        moves = []
        piece = board.board[pos]
        row, col = board.reverse_mapping[pos]

        for dr, dc in self.directions[piece]:
            new_row, new_col = row + dr, col + dc
            if board.is_valid_position(new_row, new_col):
                new_pos = board.get_position(new_row, new_col)
                if board.board[new_pos] == 0:
                    moves.append([pos, new_pos])
        
        return moves
    

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.king_value = 2.0

        # main network layers
        self.layers = nn.Sequential(
            nn.Linear(32, 40),
            nn.Tanh(),
            nn.Linear(40, 10),
            nn.Tanh(),
            nn.Linear(10, 1),
            nn.Tanh()
        )

        self.direct_connection = nn.Parameter(torch.ones(32))
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            nn.init.zeros_(module.bias)
    
    def forward(self, x):
        network_output = self.layers(x)

        direct_output = torch.sum(x * self.direct_connection, dim=1, keepdim=True)

        return network_output + direct_output
    
    def state_dict(self, *args, **kwargs):
        """Overrides state_dict to include king value"""
        state_dict = super().state_dict(*args, **kwargs)
        state_dict['king_value'] = self.king_value
        return state_dict
    
    def load_state_dict(self, state_dict, strict=True):
        """Overrides load_state_dict to handle king_value"""
        king_value = state_dict.pop('king_value', 2.0)
        super().load_state_dict(state_dict, strict=strict)
        self.king_value = king_value
    
    def evaluate(self, board: CheckersBoard) -> float:
        """Evaluate a board position"""
        input_vector = torch.tensor(board.board, dtype=torch.float32)

        input_vector[torch.abs(input_vector) > 1] *= self.king_value/2
        return input_vector.view(1, -1)
    
    def prepare_input_vector(self, board: CheckersBoard) -> torch.Tensor:
        """Converts board state into neural network input"""
        input_vector = torch.tensor(board.board, dtype=torch.float32)
        # king value multiplier
        input_vector[torch.abs(input_vector) > 1] *= self.king_value / 2
        return input_vector.view(1, -1)


class BoardEvaluator:
    def __init__(self, network: NeuralNetwork):
        self.network = network
        self.move_generator = MoveGenerator()
    
    def evaluate_position(self, board: CheckersBoard) -> float:
        """Evaluate a board position"""
        if self.is_terminal(board):
            if len(self.move_generator.get_legal_moves(board)) == 0:
                return -1.0 if board.current_player == 1 else 1.0
            return 0.0  # draw
        return self.network.evaluate(board)
    
    def is_terminal(self, board: CheckersBoard) -> bool:
        """Check if position in a move is terminal"""
        if not self.move_generator.get_legal_moves(board):
            return True
        
        # no pieces left
        red_pieces = torch.sum(torch.tensor(board.board) > 0)
        white_pieces = torch.sum(torch.tensor(board.board) < 0)
        return red_pieces == 0 or white_pieces == 0
    

class GamePlayer:
    def __init__(self, network: NeuralNetwork):
        self.network = network
        self.move_generator = MoveGenerator()
        self.evaluator = BoardEvaluator(network)
    
    def get_best_move(self, board: CheckersBoard, depth: int) -> List[int]:
        """Get best move using alpha-beta search"""
        legal_moves = self.move_generator.get_legal_moves(board)
        if not legal_moves:
            return None
        
        best_move = legal_moves[0]
        best_value = float('-inf')
        alpha = float('-inf')
        beta = float('inf')

        for move in legal_moves:
            new_board = board.copy()
            new_board.make_move(move)
            value = -self.alpha_beta(new_board, depth-1, -beta, -alpha, False)

            if value > best_value:
                best_value = value
                best_move = move

            alpha = max(alpha, value)
        
        return best_move
    
    def alpha_beta(self, board: CheckersBoard, depth: int, alpha: float, beta: float, 
    maximizing: bool) -> float:
        """Implementation of the allpha-beta search"""
        if depth == 0 or self.evaluator.is_terminal(board):
            return self.evaluator.evaluate_position(board)
        
        legal_moves = self.move_generator.get_legal_moves(board)
        if not legal_moves:
            return float('-inf') if maximizing else float('inf')
        
        if maximizing:
            value = float('-inf')
            for move in legal_moves:
                new_board = board.copy()
                new_board.make_move(move)
                value = max(value, -self.alpha_beta(new_board, depth-1, -beta, -alpha, False))
                alpha = max(alpha, value)
                if alpha >= beta:
                    break
            return value
        else:
            value = float('inf')
            for move in legal_moves:
                new_board = board.copy()
                new_board.make_move(move)
                value = max(value, -self.alpha_beta(new_board, depth-1, -beta, -alpha, True))
                beta = min(beta, value)
                if alpha >= beta:
                    break
            return value


class EvolutionaryTrainer:
    def __init__(self, population_size: int = 15, device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        self.population_size = population_size
        self.device = device
        self.population = [NeuralNetwork().to(device) for _ in range(population_size)]
        self.tournament_organizer = TournamentOrganizer(device)
        self.mutation_operator = MutationOperator(device)


    def train(self, generations: int):
        """Train the population for specified number of generations"""
        for gen in range(generations):
            logger.info(f'Generation {gen + 1}/{generations}')

            # evaluate the population
            scores = self.tournament_organizer.run_tournament(self.population)

            # select best performers
            elite_indices = torch.argsort(torch.tensor(scores))[-self.population_size // 2:]

            # create new population
            new_population = []

            # keep only elite performers
            for idx in elite_indices:
                new_population.append(self.population[idx])
            
            # create offspring through mutation
            while len(new_population) < self.population_size:
                parent = random.choice(elite_indices.tolist())
                offspring = self.mutation_operator.mutate_network(self.population[parent])
                new_population.append(offspring)
            
            self.population = new_population

    def get_best_network(self) -> NeuralNetwork:
        """Return the best performing neural network object"""
        scores = self.tournament_organizer.run_tournament(self.population)
        return self.population[torch.argmax(torch.tensor(scores))]


class MutationOperator:
    def __init__(self, device: str):
        self.device = device
        self.weight_mutation_rate = 0.1
        self.weight_mutation_range = 0.2
        self.king_value_mutation_rate = 0.1
    
    def mutate_network(self, network: NeuralNetwork) -> NeuralNetwork:
        """Creates a mutated copy of the network"""
        new_network = NeuralNetwork().to(self.device)

        # deep copy of the original network's state dict
        new_network.load_state_dict(network.state_dict())

        # mutate weights and biases
        with torch.no_grad():
            for name, param in new_network.named_parameters():
                if random.random() < self.weight_mutation_rate:
                    mutation = torch.randn_like(param) * self.weight_mutation_range
                    param.data.add_(mutation)
        
        # mutate king value
        if random.random() < self.king_value_mutation_rate:
            new_network.king_value = self.mutate_king_value(network.king_value)
        
        return new_network
    
    def mutate_king_value(self, king_value: float) -> float:
        """Mutate a king value but keep in range [1.0, 3.0]"""
        mutation = torch.normal(torch.tensor(0.0), torch.tensor(0.1)).item()
        return float(torch.clamp(torch.tensor(king_value + mutation), 1.0, 3.0))


class TournamentOrganizer:
    def __init__(self, device: str):
        self.device = device
        self.game_player = None
    
    def run_tournament(self, population: List[NeuralNetwork]) -> np.ndarray:
        """Run a tournament between different networks"""
        scores = torch.zeros(len(population), device=self.device)

        for i in range(len(population)):
            # have each network play 10 games on average
            opponents = self.select_opponents(5, len(population), exclude=i)

            for opp in opponents:
                # play two games, switching colors after the first
                result1 = self.play_game(population[i], population[opp])
                result2 = self.play_game(population[opp], population[i])

                # update scores
                scores[i] += self.calculate_score(result1)
                scores[i] += self.calculate_score(-result2)  # negate score for color switch
        
        return scores.cpu().numpy()
    
    def select_opponents(self, num_opponents: int, population_size: int, exclude: int) -> List[int]:
        """Randomly select opponents for a network"""  # TODO: is there edge in giving certain networks specific opponents?
        available = list(range(population_size))
        available.remove(exclude)
        return random.sample(available, num_opponents)

    def play_game(self, network1: NeuralNetwork, network2: NeuralNetwork) -> int:
        """Plays a game between two networks"""
        board = CheckersBoard()
        self.game_player = GamePlayer(network1)
        opponent_player = GamePlayer(network2)

        moves_without_capture = 0

        while moves_without_capture < 100:  # draw after 100 moves without capture  # TODO: coul make this time-based
            if board.current_player == 1:
                move = self.game_player.get_best_move(board, depth=4)
            else:
                move = opponent_player.get_best_move(board, depth=4)
            
            if not move:
                return -1 if board.current_player == 1 else 1
            
            # check if a move is a capture
            if abs(move[0] - move[-1]) > 4:
                moves_without_capture = 0
            else:
                moves_without_capture += 1
            
            board.make_move(move)
        
        return 0  # draw
    
    def calculate_score(self, result: int) -> float:
        """Calculates score for a game result"""
        if result == 1:  # win
            return 1.0
        elif result == 0:  # draw
            return 0.0
        else:  # loss
            return -2.0


def main():
    trainer = EvolutionaryTrainer(population_size=15)

    trainer.train(generations=250)

    best_network = trainer.get_best_network()

    torch.save({
        'model_state_dict': best_network.state_dict(),
        'king_value': best_network.king_value,
    }, 'best_checkers_network.pth')


def benchmark_ai(network_path: str, num_games: int = 100):
    """Test the AI against different strategies"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    network = NeuralNetwork().to(device)
    checkpoint = torch.load(network_path, map_location=device)
    network.load_state_dict(checkpoint['model_state_dict'])
    network.eval()

    ai_player = GamePlayer(network)
    tournament = TournamentOrganizer(device)

    # Create a simple random player for comparison
    class RandomPlayer:
        def get_best_move(self, board, depth):
            moves = MoveGenerator().get_legal_moves(board)
            return random.choice(moves) if moves else None

    results = {'wins': 0, 'losses': 0, 'draws': 0}

    for game in range(num_games):
        board = CheckersBoard()
        moves_without_capture = 0

        while moves_without_capture < 100:
            if board.current_player == 1:
                move = ai_player.get_best_move(board, depth=4)
            else:
                move = RandomPlayer().get_best_move(board, depth=1)

            if not move:
                if board.current_player == 1:
                    results['losses'] += 1
                else:
                    results['wins'] += 1
                break

            if abs(move[0] - move[-1]) > 4:
                moves_without_capture = 0
            else:
                moves_without_capture += 1

            board.make_move(move)

            if moves_without_capture >= 100:
                results['draws'] += 1

        if game % 10 == 0:
            print(f"Completed {game}/{num_games} games")

    print("\nBenchmark Results:")
    print(f"Wins: {results['wins']} ({results['wins']/num_games*100:.1f}%)")
    print(f"Losses: {results['losses']} ({results['losses']/num_games*100:.1f}%)")
    print(f"Draws: {results['draws']} ({results['draws']/num_games*100:.1f}%)")


if __name__ == '__main__':
    benchmark_ai(network_path='best_checkers_network.pth')