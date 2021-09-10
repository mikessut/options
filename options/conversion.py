from options import PutOption, CallOption

class Conversion:
  """
  Long underlying
  Long put
  Short call
  """
  
  def __init__(self, put: PutOption, call: CallOption):
    self._put = put
    self._call = call

  def value(self, price_func='mid'):
    """
    Short extrinsic value minus long intrinsic value
    """
    return self._call.extrinsic_val() - self._put.extrinsic_val()


class ReverseConversion:
  """
  Short underlying
  Long Call
  Short put
  """

  def __init__(self, put: PutOption, call: CallOption):
    self._put = put
    self._call = call

  def value(self, price_func='mid'):
    """
    Short extrinsic value minus long intrinsic value
    """
    return self._put.extrinsic_val() - self._call.extrinsic_val()