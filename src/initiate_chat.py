# from chatbot_v1.chatbot import *
from chatbot_v2.chatbot import *
from config.global_vars import *

if __name__ == "__main__":
    print(f"start talking with Lydya {txtcolor.LYDYA_VERSION}(v{VERSION}){txtcolor.ENDC}!")
    status = True
    while status:
        status = chat()