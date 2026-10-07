"""Experimental consumer-only subclass for the pinned background module.
Not integrated into Marin. Private protocol binding must be reviewed on code changes.
Polling permits stop observation; it does not cancel the producer or bound stop(join).
"""
import queue

def consumer_candidate(background_module):
    class QueueConsumerCandidate(background_module.BackgroundIterator):
        def __init__(self,*args,**kwargs):
            self._consumer_terminal=False
            super().__init__(*args,**kwargs)

        def __next__(self):
            if self.thread is None:
                if self._consumer_terminal:
                    raise StopIteration
                try:
                    return super().__next__()
                except StopIteration:
                    self._consumer_terminal=True
                    raise
            while True:
                if self._stop_event.is_set() or self._consumer_terminal:
                    raise StopIteration
                try:
                    item=self.q.get(timeout=.05)
                except queue.Empty:
                    continue
                if self._stop_event.is_set():
                    raise StopIteration
                if item is background_module._SENTINEL:
                    self._consumer_terminal=True
                    raise StopIteration
                if isinstance(item,background_module._ExceptionWrapper):
                    self._consumer_terminal=True
                    item.reraise()
                return item
    return QueueConsumerCandidate
