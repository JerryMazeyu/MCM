import datetime

def log(*args, level="INFO", **kwargs):
    valid_levels = ["INFO", "WARNING", "TOUSER", "TOLLM"]
    if level not in valid_levels:
        raise ValueError(f"Level must be one of {valid_levels}")
    
    current_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    prefix = f"[{current_time}] [{level}] \t"
    print(prefix, *args, **kwargs)

if __name__ == '__main__':
    log("hello", "world", level="WARNING")

