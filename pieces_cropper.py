import cv2 as cv

from piece_predictor import PiecePredictor

piece_predictor = PiecePredictor()


def crop_squares(board_image):
    # crop the squares from the board image
    # return a list of images of the squares
    image_width = board_image.shape[1]
    image_height = board_image.shape[0]
    squares = []
    for i in range(8):
        for j in range(8):
            y = int(i * image_width / 8)
            x = int(j * image_height / 8)
            w = int(image_width / 8)
            h = int(image_height / 8)
            square = board_image[y:y + h, x:x + w]
            squares.append(square)
    return squares


def crop_pieces(board_image):
    # crop the pieces from the board image
    # return a list of images of the pieces
    squares = crop_squares(board_image)
    # make the prediction multithreaded
    import concurrent.futures
    with concurrent.futures.ThreadPoolExecutor() as executor:
        predictions = executor.map(piece_predictor.predict, squares)
    return list(predictions)


FEN_PIECES = {
    'black_pawn': 'p',
    'black_knight': 'n',
    'black_bishop': 'b',
    'black_rook': 'r',
    'black_queen': 'q',
    'black_king': 'k',
    'white_pawn': 'P',
    'white_knight': 'N',
    'white_bishop': 'B',
    'white_rook': 'R',
    'white_queen': 'Q',
    'white_king': 'K',
}


def get_fen(pieces):
    # convert the list of pieces to a FEN string
    # return the FEN string
    fen = ''
    for i in range(8):
        empty_squares = 0
        for j in range(8):
            piece = pieces[i * 8 + j]
            if piece == 'empty':
                empty_squares += 1
            else:
                if empty_squares > 0:
                    fen += str(empty_squares)
                    empty_squares = 0
                fen += FEN_PIECES[piece]
        if empty_squares > 0:
            fen += str(empty_squares)
        if i < 7:
            fen += '/'
    return fen


def get_fen_from_image(board_image):
    # get the FEN string from an image of a chess board
    # return the FEN string
    pieces = crop_pieces(board_image)
    fen = get_fen(pieces)
    return fen


if __name__ == '__main__':
    board_image = cv.imread('board.png')
    fen = get_fen_from_image(board_image)
    print(fen)
