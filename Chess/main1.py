from game import Game
from search import Searcher  # Đổi tên searcher_logic thành search


def main(): 
    g = Game(None, True, True, True, True, True, 0, [])
    g.initial_pos()  # Gọi sau khi tạo đối tượng

    s = Searcher()

    while True:
        try:
            command = input()
        except EOFError:  # Thoát nếu không còn input (ví dụ khi GUI đóng pipe)
            break

        if not command.strip(): continue  # Bỏ qua dòng trống


        if command == 'uci':
            print('id author YourName')
            print('uciok')
        elif command == 'isready':
            print('readyok')
        elif command.startswith('go'):
            time_remaining = 30000.0
            c_split = command.split()
            for idx, val_cmd in enumerate(c_split):

                if val_cmd == 'wtime' and g.side:
                    time_remaining = float(c_split[idx + 1])
                elif val_cmd == 'btime' and not g.side:
                    time_remaining = float(c_split[idx + 1])
            s.search_iterative(g, time_remaining)  # time_remaining là ms

        elif command.startswith('position'):
            g.uci_position(command)
        elif command == 'quit':
            return  # Thoát vòng lặp và kết thúc chương trình


if __name__ == '__main__':
    main()