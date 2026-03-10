import streamlit.web.cli as stcli
import os, sys

def resolve_path(path):
    if getattr(sys, "frozen", False):
        basedir = sys._MEIPASS
    else:
        basedir = os.path.dirname(__file__)
    return os.path.join(basedir, path)

if __name__ == "__main__":
    # Remove the script name from argv so streamlit doesn't get confused
    # sys.argv[0] is the script name
    
    app_path = resolve_path("app.py")
    
    # Construct the arguments for streamlit
    # equivalent to: streamlit run app.py --global.developmentMode=false
    sys.argv = [
        "streamlit",
        "run",
        app_path,
        "--global.developmentMode=false",
    ]
    
    sys.exit(stcli.main())
