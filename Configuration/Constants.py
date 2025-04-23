from enum import Enum


class TimeFrames(Enum):
    H1 = "H1"
    H4 = "H4"
    D1 = "D1"


class CurrencyPairs:
    XAUUSD = "XAUUSD"
    USDJPY = "USDJPY"
    EURUSD = "EURUSD"
    GBPUSD = "GBPUSD"

    @staticmethod
    def display_name(pair_code):
        display_map = {
            "XAUUSD": "XAU - USD",
            "USDJPY": "USD - JPY",
            "EURUSD": "EUR - USD",
            "GBPUSD": "GBP - USD"
        }
        return display_map.get(pair_code, pair_code)

    @staticmethod
    def code_from_display(display_name):
        code_map = {
            "XAU - USD": "XAUUSD",
            "USD - JPY": "USDJPY",
            "EUR - USD": "EURUSD",
            "GBP - USD": "GBPUSD"
        }
        return code_map.get(display_name, display_name)
