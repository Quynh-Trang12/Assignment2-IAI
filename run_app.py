import sys
from src.core.search import SearchCLI
from src.ui.main_gui import launch_gui

if __name__ == "__main__":
    if len(sys.argv) < 3 and "--gui" not in sys.argv:
        print("Usage (Terminal): python run_app.py <filepath> <method> [--traffic]")
        print("Usage (GUI):      python run_app.py <filepath> --gui")
        sys.exit(1)

    filepath = sys.argv[1]

    if "--gui" in sys.argv:
        launch_gui(filepath)
    else:
        SearchCLI.execute()
