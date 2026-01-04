import logging
from termcolor import colored
from logging import handlers


class Logger(object):
    level_relations = {
        'debug': logging.DEBUG,
        'info': logging.INFO,
        'warning': logging.WARNING,
        'error': logging.ERROR,
        'crit': logging.CRITICAL
    }


    # 日志级别关系映射
    def __init__(self, filename, level='info', when='D', backCount=3,
                 fmt='%(asctime)s - %(levelname)s: %(message)s'):
        color_fmt = colored('[%(asctime)s %(name)s]', 'green') + colored('(%(filename)s %(lineno)d)', 'yellow') + ': %(levelname)s %(message)s'
        self.filename = filename
        self.level = level
        self.fmt = fmt

        self.logger = logging.getLogger(self.filename)
        self.logger.propagate = False  # 防止终端重复打印


        # 设置日志级别
        self.logger.setLevel(self.level_relations.get(self.level))

        if not self.logger.handlers:
            # self.logger.handlers.clear()

            # 设置日志格式
            print_format_str = logging.Formatter(fmt=color_fmt)

            # 设置屏幕打印
            sh = logging.StreamHandler()  # 往屏幕上输出

            # 设置打印格式
            sh.setFormatter(print_format_str)  # 设置屏幕上显示的格式

            # 把对象加到logger里
            self.logger.addHandler(sh)

            # 往文件里写入handler
            # th = handlers.TimedRotatingFileHandler(filename=filename, backupCount=backCount, encoding='utf-8')
            th = logging.FileHandler(filename, 'a', encoding='utf-8')

            log_format_str = logging.Formatter(self.fmt)
            # 设置文件里写入的格式
            th.setFormatter(log_format_str)

            # 添加写入文件的操作
            self.logger.addHandler(th)

            # 实例化TimedRotatingFileHandler
            # interval是时间间隔，backupCount是备份文件的个数，如果超过这个个数，就会自动删除，when是间隔的时间单位，单位有以下几种：
            # S 秒
            # M 分
            # H 小时、
            # D 天、
            # W 每星期（interval==0时代表星期一）
            # midnight 每天凌晨
        # return self.logger


if __name__ == '__main__':
    log = Logger('all.log', level='info')
    # log.logger.debug('debug')

    log.logger.info("train_acc:{} train_loss:{}".format(91.2, 0.34))
    log.logger.info("train_acc:{} train_loss:{}".format(9.2, 0.3))
    log.logger.info("train_acc:{} train_loss:{}".format(921.2, 0.4))
    log.logger.info("train_acc:{} train_loss:{}".format(1.2, 0.334))
    # log.logger.warning('警告')
    # log.logger.error('报错')
    # log.logger.critical('严重')
    # Logger('error.log', level='error').logger.error('error')


