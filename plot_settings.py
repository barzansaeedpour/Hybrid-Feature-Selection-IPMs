

class PlotSettings:
    current_marker_index = 0
    colors = ['cyan','purple', 'black', 'blue', 'orange', 'lightgray', 'lime','seagreen','navy','slateblue','teal','olive','navajowhite','tan','red','yellow',]
    markers=['.', 'o', 'v', '^', '<', '>', 's', 'p', '*', 'h', 'H', '+', 'x', 'D', 'd', '|', '_', 'P', 'X', 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11,]
    
    def __init__(self) -> None:
        pass

    def next_marker(self):
        PlotSettings.current_marker_index = PlotSettings.current_marker_index + 1