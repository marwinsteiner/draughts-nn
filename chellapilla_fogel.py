from types import new_class
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import List, Tuple, Optional
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
        return self.get_legal_moves(board)
    
    def get_all_jumps(self, board: CheckersBoard) -> List[List[int]]:
        """Get all possible jump moves"""
        all_jumps = []
        for pos in range(32):
            if board.board[pos] * board.current_player > 0:
                jumps = self.get_piece_jumps(board, pos, [pos])
                all_jumps.extend(jumps)
        return all_jumps
    
    def get_piece_jumps(self, board: CheckersBoard, pos: int,
    current_sequence: List[int]) -> List[Liat[int]]:
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

            if (middle_pos is not None and jump_pos is not None and board.board[middle_pos] * board-current_player < 0 and board.board[jump_pos] == 0):
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
            if board.board[pos] * board.current > 0:
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
    
    def evaluate(self, board: CheckersBoard) -> float:
        """Evaluate a board position"""
        input_vector = torch.tensor(board.board, dtype=torch.float32)

        input_vector[torch.abs(input_vector) > 1] *= self.king_value/2
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
    
