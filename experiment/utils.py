import os

from line_notify import LineNotify

ACCESS_TOKEN = open(f"{os.path.dirname(__file__)}/../LINE_TOKEN.txt").readlines()[0].strip();


def notify_line(template: str) -> None:
    """
    Notify training results to LINE via token
    
    Argument
    --------
    template: str
        String template format to send to
    """
    if ACCESS_TOKEN is None or ACCESS_TOKEN == "":
        return
    notify: LineNotify = LineNotify(ACCESS_TOKEN);
    notify.send("\n\n" + template);
