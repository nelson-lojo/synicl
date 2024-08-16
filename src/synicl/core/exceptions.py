
class AbstractionError(Exception):
    '''An abstraction barrier has been violated!'''

class UnexpectedError(Exception):
    '''A marker to indicate a code path *should* be unreachable'''

class ShapeError(Exception):
    '''Wrong Shapes encountered!'''

class UninitializedError(Exception):
    '''A behavior was attempted without initializing the necessary object!'''

class IncompleteImplementationError(Exception):
    '''An extending implementation failed to follow conventions!'''
