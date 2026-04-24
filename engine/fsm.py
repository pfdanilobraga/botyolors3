from transitions import Machine

class BotEngineFSM:
    """
    Base Finite State Machine for the Bot Engine.
    Handles states and valid transitions.
    """
    # The states are defined in the specific implementation classes (like StateMining)
    # but we initialize the machine here.
    
    def __init__(self, states, initial_state):
        self.machine = Machine(
            model=self,
            states=states,
            initial=initial_state,
            send_event=True, # passes the event data to the callbacks
            queued=True # Avoids recursion depth exhaustion on loops
        )
