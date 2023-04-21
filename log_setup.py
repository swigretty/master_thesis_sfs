import copy
import json
import logging.config
from pathlib import Path
from logging import getLogger


# uses module-level loggers by passing __name__ as the name parameter to getLogger()
# to create a logger object as the name of the logger itself would tell us from where
# the events are being logged. (__name__ evaluates to the current module, or __main__ module name
# where execution starts)


_LOGGING = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'verbose': {
            'format': '%(asctime)s %(name)s:%(levelname)s %(filename)s:'
                      '%(lineno)s:%(funcName)s(): %(message)s'
        },
        'verbose-rsyslog': {
            'format': 'send_mail:%(process)d:%(name)s:%(levelname)s '
                      '%(filename)s:%(lineno)s:%(funcName)s(): %(message)s',
        },
    },
    # Handlers send the LogRecord to the required output destination, like the console or a file
    'handlers': {
        'stderr': {
            # don't log below warning to stderr
            'class': 'logging.StreamHandler',
            'formatter': 'verbose',
            'level': 'INFO',
            'stream': "ext://sys.stderr",
        },
        'stdout': {
            'class': 'logging.StreamHandler',
            'formatter': 'verbose',
            'stream': "ext://sys.stdout",
        },
        # 'rsyslog': {
        #     'level': 'WARNING',
        #     'class': 'logging.handlers.SysLogHandler',
        #     'formatter': 'verbose-rsyslog',
        #     'address': '/dev/log',  # use Syslog's default unix socket
        #     'facility': 'user',
        #     # choose a facility that is configured either in
        #     # /etc/rsyslog.conf or /etc/rsyslog.d/*
        #     # 'user' is chosen by default
        # },
        'nullhandler': {
            'class': 'logging.NullHandler',
        }
    },
    'loggers': {
        'master_thesis_sfs': {
            # A logger can have more then oune handler. Each handler can have its own (1) formatter
            'handlers': ['stdout'],
            'level': 'INFO',
            'propagate': False,
            # propagate = True. events logged to this logger will be passed to the handlers of
            # higher level (ancestor) logger, in addition to any handlers attached to this logger
            # A common scenario is to attach handlers only to the root logger,
            # and to let propagation take care of the rest.
            # root = logging.getLogger()
            # root.addHandler(null_handler)
        },
        '__main__': {
            'handlers': ['stderr'],
            'level': 'INFO',
            'propagate': False,
        }
    }

}


def get_config_dict(custom_log_json='.custom_logging_color.json'):
    config_dict = _LOGGING

    if isinstance(custom_log_json, dict):
        config_dict = custom_log_json

    if Path(custom_log_json_path := Path(custom_log_json)).exists():
        pass
    elif Path(custom_log_json_path := Path.home() / custom_log_json).exists():
        pass
    elif Path(custom_log_json_path := Path(__file__).resolve().parent / custom_log_json).exists():
        pass
    else:
        custom_log_json_path = None
        print(f"Custom logging: {custom_log_json} could not be found. Default logging used")

    if custom_log_json_path is not None:
        with open(custom_log_json_path, 'r') as f:
            config_dict = json.load(f)

    return config_dict


def setup_logging(custom_log_json='.custom_logging_color.json'):
    logging.config.dictConfig(get_config_dict(custom_log_json=custom_log_json))


if __name__ == '__main__':
    logger = getLogger(__name__)
    setup_logging()
    logger.info('This is a info message', extra={"addtional_info": "This is a test"})
    logger.debug('This is a debug message', extra={"addtional_info": "This is a test"})
