from splendor.exceptions import SplendorContextException

_context_state = {
    'initialized' : False,
    'mode' : None,
}

def initialization_state():
    return _context_state['initialized'], _context_state['mode']

def register_context(context_mode):
    if not _context_state['initialized']:
        _context_state['initialized'] = True
        _context_state['mode'] = context_mode
        return True
    elif _context_state['mode'] == context_mode:
        return False
    else:
        raise SplendorContextException(
                'Cannot register a new context, one already exists (%s)'%
                (_context_state['mode']))
