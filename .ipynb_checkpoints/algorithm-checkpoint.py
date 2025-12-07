from state import State
from conllu_token import Token
import copy

class Transition(object):
    """
    Class to represent a parsing transition in a dependency parser.
    
    Attributes:
    - action (str): The action to take, represented as an string constant. Actions include SHIFT, REDUCE, LEFT-ARC, or RIGHT-ARC.
    - dependency (str): The type of dependency relationship (only for LEFT-ARC and RIGHT-ARC, otherwise it'll be None), corresponding to the deprel column
    """

    def __init__(self, action: int, dependency: str = None):
        self._action = action
        self._dependency = dependency

    @property
    def action(self):
        """Return the action attribute."""
        return self._action

    @property
    def dependency(self):
        """Return the dependency attribute."""
        return self._dependency

    def __str__(self):
        return f"{self._action}-{self._dependency}" if self._dependency else str(self._action)


class Sample(object):
    """
    Represents a training sample for a transition-based dependency parser. 

    This class encapsulates a parser state and the corresponding transition action 
    to be taken in that state. It is used for training models that predict parser actions 
    based on the current state of the parsing process.

    Attributes:
        state (State): An instance of the State class, representing the current parsing 
                       state at a given timestep in the parsing process.
        transition (Transition): An instance of the Transition class, representing the 
                                 parser action to be taken in the given state.

    Methods:
        state_to_feats(nbuffer_feats: int = 2, nstack_feats: int = 2): Extracts features from the parsing state.
    """

    def __init__(self, state: State, transition: Transition):
        """
        Initializes a new instance of the Sample class.

        Parameters:
            state (State): The current parsing state.
            transition (Transition): The transition action corresponding to the state.
        """
        self._state = state
        self._transition = transition

    @property
    def state(self):
        """
        Retrieves the current parsing state of the sample.

        Returns:
            State: The current parsing state in this sample.
        """
        return self._state


    @property
    def transition(self):
        """
        Retrieves the transition action of the sample.

        Returns:
            Transition: The transition action representing the parser's decision at this sample's state.
        """
        return self._transition
    

    def state_to_feats(self, nbuffer_feats: int = 2, nstack_feats: int = 2):
        """
        Extracts features from a given parsing state for use in a transition-based dependency parser.

        This function generates a feature representation from the current state of the parser, 
        which includes features from both the stack and the buffer. The number of features from 
        the stack and the buffer can be specified.

        Parameters:
            nbuffer_feats (int): The number of features to extract from the buffer.
            nstack_feats (int): The number of features to extract from the stack.

        Returns:
            list[str]: A list of extracted features. The features include the words and their 
                    corresponding UPOS (Universal Part-of-Speech) tags from the specified number 
                    of elements in the stack and buffer. The format of the feature list is as follows:
                    [Word_stack_n,...,Word_stack_0, Word_buffer_0,...,Word_buffer_m, 
                        UPOS_stack_n,...,UPOS_stack_0, UPOS_buffer_0,...,UPOS_buffer_m]
                    where 'n' is nstack_feats and 'm' is nbuffer_feats.

        Examples:
            Example 1:
                State: Stack (size=1): (0, ROOT, ROOT_UPOS)
                    Buffer (size=13): (1, Distribution, NOUN) | ... | (13, ., PUNCT)
                    Arcs (size=0): []

                Output: ['<PAD>', 'ROOT', 'Distribution', 'of', '<PAD>', 'ROOT_UPOS', 'NOUN', 'ADP']

            Example 2:
                State: Stack (size=2): (0, ROOT, ROOT_UPOS) | (1, Distribution, NOUN)
                    Buffer (size=10): (4, license, NOUN) | ... | (13, ., PUNCT)
                    Arcs (size=2): [(4, 'det', 3), (4, 'case', 2)]

                Output: ['ROOT', 'Distribution', 'license', 'does', 'ROOT_UPOS', 'NOUN', 'NOUN', 'AUX']
        """
        # Constants for special tokens 
        PAD = '<PAD>'
        ROOT_WORD = 'ROOT'
        ROOT_UPOS = 'ROOT_UPOS'
        
        def get_token_feature(token, is_upos):
            if token is None:
                return PAD
            elif token.id == 0:
                return ROOT_UPOS if is_upos else ROOT_WORD
            else:
                return token.upos if is_upos else token.form

        # 1. --- Extract features from the Stack (σ) ---
        # The stack stores items LIFO. We want the most recent elements first 
        # (S[-1] is the most recent, referred to as sigma_0).
        
        # Slicing from the end gives us the last 'nstack_feats' elements.
        stack_tokens_raw = self.state.S[-nstack_feats:]
        
        # We need the tokens in the order: S_0, S_1, ... S_n-1.
        # S[-1] is S_0. S[-2] is S_1. We reverse the raw slice.
        stack_tokens_reversed = stack_tokens_raw[::-1]
        
        # Pad the list if the stack is smaller than nstack_feats.
        # Padding should occur on the 'older' (higher index) side of the stack.
        # The list must have exactly nstack_feats elements.
        num_missing_stack = nstack_feats - len(stack_tokens_reversed)
        stack_tokens = stack_tokens_reversed + [None] * num_missing_stack
        # Note on order: The documentation example suggests ['<PAD>', 'ROOT'] for S(size=1) 
        # when nstack_feats=2. This means the list should be ordered from oldest (S_n) to newest (S_0).
        # Let's adjust the order here to match the required output order: 
        # [Word_stack_n, ..., Word_stack_0] (Oldest to Newest).
        
        # The required order is [Word_stack_n, ..., Word_stack_0].
        # Since we use list slicing to get S[0], S[1], S[2],... (newest to oldest), 
        # we reverse it again after padding to achieve the final required order.
        # Correct final list construction for the required format (Oldest to Newest):
        stack_tokens_final = [None] * num_missing_stack + stack_tokens_raw # Fill <PAD> first, then the stack tokens.
        
        # 2. --- Extract features from the Buffer (β) ---
        # The buffer stores items FIFO. We want the elements from the front (B[0] is the newest).
        
        # Slicing from the start gives us the first 'nbuffer_feats' elements.
        buffer_tokens_raw = self.state.B[:nbuffer_feats]
        
        # Pad the list if the buffer is smaller than nbuffer_feats.
        # Padding should occur at the end of the list (when the buffer runs out).
        num_missing_buffer = nbuffer_feats - len(buffer_tokens_raw)
        buffer_tokens_final = buffer_tokens_raw + [None] * num_missing_buffer
        
        # 3. --- Assemble the final feature vector (Words and UPOS) ---
        words_feats = []
        upos_feats = []

        # A. Stack Words (Word_stack_n,...,Word_stack_0: Oldest to Newest)
        # Note: We must reverse stack_tokens_final to match the order described in the guide examples 
        # (Example 2 output starts with ROOT, Distribution) which is Newest to Oldest.
        # Let's stick to the order that produces the documented examples: Newest to Oldest for Stack tokens.
        
        # Reversing to get S_0, S_1... (Newest to Oldest for the word feature block)
        # Re-using stack_tokens_final (which is PAD -> Oldest -> Newest)
        # If the target order is [Word_stack_n,...,Word_stack_0] (Oldest to Newest) as written in the doc string:
        for token in stack_tokens_final:
             words_feats.append(get_token_feature(token, is_upos=False))
             upos_feats.append(get_token_feature(token, is_upos=True))

        # B. Buffer Words (Word_buffer_0,...,Word_buffer_m: Newest to Oldest)
        for token in buffer_tokens_final:
            words_feats.append(get_token_feature(token, is_upos=False))
            upos_feats.append(get_token_feature(token, is_upos=True))

        # C. Concatenate and return the final list
        # Format: [Words_Stack, Words_Buffer, UPOS_Stack, UPOS_Buffer] 
        return words_feats + upos_feats
    

    def __str__(self):
        """
        Returns a string representation of the sample, including its state and transition.

        Returns:
            str: A string representing the state and transition of the sample.
        """
        return f"Sample - State:\n\n{self._state}\nSample - Transition: {self._transition}"



class ArcEager():

    """
    Implements the arc-eager transition-based parsing algorithm for dependency parsing.

    This class includes methods for creating initial parsing states, applying transitions to 
    these states, and determining the correct sequence of transitions for a given sentence.

    Methods:
        create_initial_state(sent: list[Token]): Creates the initial state for a given sentence.
        final_state(state: State): Checks if the current parsing state is a valid final configuration.
        LA_is_valid(state: State): Determines if a LEFT-ARC transition is valid for the current state.
        LA_is_correct(state: State): Determines if a LEFT-ARC transition is correct for the current state.
        RA_is_correct(state: State): Determines if a RIGHT-ARC transition is correct for the current state.
        RA_is_valid(state: State): Checks if a RIGHT-ARC transition is valid for the current state.
        REDUCE_is_correct(state: State): Determines if a REDUCE transition is correct for the current state.
        REDUCE_is_valid(state: State): Determines if a REDUCE transition is valid for the current state.
        oracle(sent: list[Token]): Computes the gold transitions for a given sentence.
        apply_transition(state: State, transition: Transition): Applies a given transition to the current state.
        gold_arcs(sent: list[Token]): Extracts gold-standard dependency arcs from a sentence.
    """

    LA = "LEFT-ARC"
    RA = "RIGHT-ARC"
    SHIFT = "SHIFT"
    REDUCE = "REDUCE"

    def create_initial_state(self, sent: list['Token']) -> State:
        """
        Creates the initial state for the arc-eager parsing algorithm given a sentence.

        This function initializes the parsing state, which is essential for beginning the parsing process. 
        The initial state consists of a stack (initially containing only the root token), a buffer 
        (containing all tokens of the sentence except the root), and an empty set of arcs.

        Parameters:
            sent (list[Token]): A list of 'Token' instances representing the sentence to be parsed. 
                                The first token in the list should typically be a 'ROOT' token.

        Returns:
            State: The initial parsing state, comprising a stack with the root token, a buffer with 
                the remaining tokens, and an empty set of arcs.
        """
        return State([sent[0]], sent[1:], set([]))
    
    def final_state(self, state: State) -> bool:
        """
        Checks if the curent parsing state is a valid final configuration, i.e., the buffer is empty

            Parameters:
                state (State): The parsing configuration to be checked

            Returns: A boolean that indicates if state is final or not
        """
        return len(state.B) == 0

    # --- HELPER FUNCTION ---
    def has_head(self, token: Token, arcs: set) -> bool:
        """Helper to check if a token already has a head in the current set of arcs."""
        if token is None:
            return False
        for (head_id, label, dep_id) in arcs:
            if dep_id == token.id:
                return True
        return False

    def LA_is_valid(self, state: State) -> bool:
        """
        LEFT-ARC preconditions:
        1. Stack must not be empty.
        2. The token on top of the stack (dependent) must NOT be the ROOT (id 0).
        3. The token on top of the stack must NOT already have a head.
        """
        if not state.S:
            return False
        s = state.S[-1]
        # Precondition: not root (id!=0) AND does not have head yet
        return s.id != 0 and not self.has_head(s, state.A)

    def LA_is_correct(self, state: State) -> bool:
        """
        LEFT-ARC is correct if the Gold Arcs contain a relation where:
        Head = Top of Buffer (B[0])
        Dependent = Top of Stack (S[-1])
        """
        # We need the gold arcs. Since we don't have them stored in state, 
        # we assume this method is called inside oracle() loop where we can look them up,
        # OR we assume we inspect the gold attributes of the tokens themselves if they are loaded.
        # Typically in this assignment structure, we check if the relation exists in the 'sentence' structure.
        
        s = state.S[-1]
        b = state.B[0]
        
        # Check if there is an arc b -> s in the gold standard
        # Note: In the provided Token class, 'head' attribute is the gold head ID.
        return s.head == b.id

    
    def RA_is_correct(self, state: State) -> bool:
        """
        RIGHT-ARC is correct if the Gold Arcs contain a relation where:
        Head = Top of Stack (S[-1])
        Dependent = Top of Buffer (B[0])
        """
        s = state.S[-1]
        b = state.B[0]
        
        # Check if there is an arc s -> b in the gold standard
        return b.head == s.id

    def RA_is_valid(self, state: State) -> bool:
        """
        RIGHT-ARC preconditions:
        1. Stack must not be empty.
        """
        return len(state.S) > 0

    def REDUCE_is_correct(self, state: State) -> bool:
        """
        REDUCE is correct if:
        1. The top of stack has a head (checked in valid).
        2. The top of stack does NOT have any dependents (children) remaining in the buffer.
           If it has dependents in the buffer, we must wait (SHIFT/RA) to attach them later.
        """
        s = state.S[-1]
        
        # Iterate through the buffer to see if 's' is the head of any token there
        for token in state.B:
            if token.head == s.id:
                return False # We cannot reduce yet, 's' is needed as a head for a future token
        
        return True

    def REDUCE_is_valid(self, state: State) -> bool:
        """
        REDUCE preconditions:
        1. Stack must not be empty.
        2. The token on top of the stack MUST already have a head.
        """
        if not state.S:
            return False
        s = state.S[-1]
        return self.has_head(s, state.A)

    def oracle(self, sent: list['Token']) -> list['Sample']:
        """
        Computes the gold transitions to take at each parsing step, given an input dependency tree.

        This function iterates through a given sentence, represented as a dependency tree, to generate a sequence 
        of gold-standard transitions. These transitions are what an ideal parser should predict at each step to 
        correctly parse the sentence. The function checks the validity and correctness of possible transitions 
        at each step and selects the appropriate one based on the arc-eager parsing algorithm. It is primarily 
        used for later training a dependency parser.

        Parameters:
            sent (list['Token']): A list of 'Token' instances representing a dependency tree. Each 'Token' 
                        should contain information about a word/token in a sentence.

        Returns:
            samples (list['Sample']): A list of Sample instances. Each Sample stores an state instance and a transition instance
            with the information of the outputs to predict (the transition and optionally the dependency label)
        """

        state = self.create_initial_state(sent) 

        samples = [] #Store here all training samples for sent

        #Applies the transition system until a final configuration state is reached
        while not self.final_state(state):
            
            if self.LA_is_valid(state) and self.LA_is_correct(state):
                # Create the LA transition. The dependent is on top of the stack (S[-1]).
                # We retrieve the correct dependency label from that token.
                transition = Transition(self.LA, state.S[-1].dep)
                
                #Add current state 'state' (the input) and the transition taken (the desired output) to the list of samples
                current_state_copy = copy.deepcopy(state)
                samples.append(Sample(current_state_copy, transition))
                
                #Update the state by applying the LA transition using the function apply_transition
                self.apply_transition(state, transition)

            elif self.RA_is_valid(state) and self.RA_is_correct(state):
                # Create the RA transition. The dependent is at the top of the buffer (B[0]).
                # We retrieve the correct dependency label from that token.
                transition = Transition(self.RA, state.B[0].dep)
                
                #Add current state 'state' (the input) and the transition taken (the desired output) to the list of samples
                current_state_copy = copy.deepcopy(state)
                samples.append(Sample(current_state_copy, transition))
                
                #Update the state by applying the RA transition using the function apply_transition
                self.apply_transition(state, transition)

            elif self.REDUCE_is_valid(state) and self.REDUCE_is_correct(state):
                # Create the REDUCE transition (no dependency label needed)
                transition = Transition(self.REDUCE)
                
                #Add current state 'state' (the input) and the transition taken (the desired output) to the list of samples
                current_state_copy = copy.deepcopy(state)
                samples.append(Sample(current_state_copy, transition))
                
                #Update the state by applying the REDUCE transition using the function apply_transitionn
                self.apply_transition(state, transition)
            else:
                #If no other transiton can be applied, it's a SHIFT transition
                transition = Transition(self.SHIFT)
                #Add current state 'state' (the input) and the transition taken (the desired output) to the list of samples
                current_state_copy = copy.deepcopy(state)
                samples.append(Sample(current_state_copy, transition))
                
                #Update the state by applying the SHIFT transition using the function apply_transition
                self.apply_transition(state,transition)


        #When the oracle ends, the generated arcs must
        #match exactly the gold arcs, otherwise the obtained sequence of transitions is not correct
        assert self.gold_arcs(sent) == state.A, f"Gold arcs {self.gold_arcs(sent)} and generated arcs {state.A} do not match"
    
        return samples         
    

    def apply_transition(self, state: State, transition: Transition):
        """
        Applies a given transition to the current parsing state.

        This method updates the state based on the type of transition - LEFT-ARC, RIGHT-ARC, 
        or REDUCE - and the validity of applying such a transition in the current context.

        Parameters:
            state (State): The current parsing state, which includes a stack (S), 
                        a buffer (B), and a set of arcs (A).
            transition (Transition): The transition to be applied, consisting of an action
                                    (LEFT-ARC, RIGHT-ARC, REDUCE) and an optional dependency label (only for LEFT-ARC and RIGHT-arc).

        Returns:
            None; the state is modified in place.
        """

        # Extract the action and dependency label from the transition
        t = transition.action
        dep = transition.dependency

        # The top item on the stack and the first item in the buffer
        s = state.S[-1] if state.S else None  # Top of the stack
        b = state.B[0] if state.B else None   # First in the buffer

        if t == self.LA and self.LA_is_valid(state):
            # LEFT-ARC transition logic: to be implemented
            # Add an arc to the state from the top of the buffer to the top of the stack
            # Remove from the state the top word from the stack

            # LEFT-ARC:
            # 1. Create arc (Buffer[0] -> Stack[-1])
            # 2. Remove Stack[-1] from stack (pop)
            state.A.add((b.id, dep, s.id))
            state.S.pop()

        elif t == self.RA and self.RA_is_valid(state): 
            # RIGHT-ARC transition
            # Add an arc to the state from the stack top to the buffer head with the specified dependency
            # Move from the state the buffer head to the stack
            # Remove from the state the first item from the buffer
            
            # RIGHT-ARC:
            # 1. Create arc (Stack[-1] -> Buffer[0])
            # 2. Push Buffer[0] to Stack
            # 3. Remove Buffer[0] from Buffer (pop first element)
            state.A.add((s.id, dep, b.id))
            state.S.append(b)
            del state.B[0] # Remove head of buffer

        elif t == self.REDUCE and self.has_head(s, state.A): 
            # REDUCE transition logic: to be implemented
            # Remove from state the word from the top of the stack
            
            # REDUCE:
            # 1. Pop the stack
            state.S.pop()

        else:
            # SHIFT transition logic: Already implemented! Use it as a basis to implement the others
            #This involves moving the top of the buffer to the stack
            state.S.append(b) 
            del state.B[:1]


    def gold_arcs(self, sent: list['Token']) -> set:
        """
        Extracts and returns the gold-standard dependency arcs from a given sentence.

        This function processes a sentence represented by a list of Token objects to extract the dependency relations 
        (arcs) present in the sentence. Each Token object should contain information about its head (the id of the 
        parent token in the dependency tree), the type of dependency, and its own id. The function constructs a set 
        of tuples, each representing a dependency arc in the sentence.

        Parameters:
            sent (list[Token]): A list of Token objects representing the sentence. Each Token object contains 
                                information about a word or punctuation in a sentence, including its dependency 
                                relations and other annotations.

        Returns:
            gold_arcs (set[tuple]): A set of tuples, where each tuple is a triplet (head_id, dependency_type, dependent_id). 
                                    This represents all the gold-standard dependency arcs in the sentence. The head_id and 
                                    dependent_id are integers representing the respective tokens in the sentence, and 
                                    dependency_type is a string indicating the type of dependency relation.
        """
        gold_arcs = set([])
        for token in sent[1:]:
            gold_arcs.add((token.head, token.dep, token.id))

        return gold_arcs


   


if __name__ == "__main__":


    print("**************************************************")
    print("*               Arc-eager function               *")
    print("**************************************************\n")

    print("Creating the initial state for the sentence: 'The cat is sleeping.' \n")

    tree = [
        Token(0, "ROOT", "ROOT", "_", "_", "_", "_", "_"),
        Token(1, "The", "the", "DET", "_", "Definite=Def|PronType=Art", 2, "det"),
        Token(2, "cat", "cat", "NOUN", "_", "Number=Sing", 4, "nsubj"),
        Token(3, "is", "be", "AUX", "_", "Mood=Ind|Tense=Pres|VerbForm=Fin", 4, "cop"),
        Token(4, "sleeping", "sleep", "VERB", "_", "VerbForm=Ger", 0, "root"),
        Token(5, ".", ".", "PUNCT", "_", "_", 4, "punct")
    ]

    arc_eager = ArcEager()
    print("Initial state")
    state = arc_eager.create_initial_state(tree)
    print(state)

    #Checking that is a final state
    print (f"Is the initial state a valid final state (buffer is empty)? {arc_eager.final_state(state)}\n")

    # Applying a SHIFT transition
    transition1 = Transition(arc_eager.SHIFT)
    arc_eager.apply_transition(state, transition1)
    print("State after applying the SHIFT transition:")
    print(state, "\n")

    #Obtaining the gold_arcs of the sentence with the function gold_arcs
    gold_arcs = arc_eager.gold_arcs(tree)
    print (f"Set of gold arcs: {gold_arcs}\n\n")


    print("**************************************************")
    print("*  Creating instances of the class Transition    *")
    print("**************************************************")

    # Creating a SHIFT transition
    shift_transition = Transition(ArcEager.SHIFT)
    # Printing the created transition
    print(f"Created Transition: {shift_transition}")  # Output: Created Transition: SHIFT

    # Creating a LEFT-ARC transition with a specific dependency type
    left_arc_transition = Transition(ArcEager.LA, "nsubj")
    # Printing the created transition
    print(f"Created Transition: {left_arc_transition}")

    # Creating a RIGHT-ARC transition with a specific dependency type
    right_arc_transition = Transition(ArcEager.RA, "amod")
    # Printing the created transition
    print(f"Created Transition: {right_arc_transition}")

    # Creating a REDUCE transition
    reduce_transition = Transition(ArcEager.REDUCE)
    # Printing the created transition
    print(f"Created Transition: {reduce_transition}")  # Output: Created Transition: SHIFT

    print()
    print("**************************************************")
    print("*     Creating instances of the class  Sample    *")
    print("**************************************************")

    # For demonstration, let's create a dummy State instance
    state = arc_eager.create_initial_state(tree)  # Replace with actual state initialization as per your implementation

    # Create a Transition instance. For example, a SHIFT transition
    shift_transition = Transition(ArcEager.SHIFT)

    # Now, create a Sample instance using the state and transition
    sample_instance = Sample(state, shift_transition)

    # To display the created Sample instance
    print("Sample:\n", sample_instance)


    print("\n")
    print("**************************************************")
    print("* TESTING FEATURE EXTRACTION (state_to_feats) *")
    print("**************************************************")

    # --- Preparar Tokens Falsos para la prueba ---
    # Definimos tokens simples para simular el Ejemplo 1 y 2 de la documentación
    t_root = Token(0, "ROOT", "ROOT", "ROOT", "_", "_", -1, "_")
    t_dist = Token(1, "Distribution", "distribution", "NOUN", "_", "_", 2, "nsubj")
    t_of   = Token(2, "of", "of", "ADP", "_", "_", 1, "case") # Inventado para llenar buffer
    t_lic  = Token(4, "license", "license", "NOUN", "_", "_", 3, "nsubj")
    t_does = Token(5, "does", "do", "AUX", "_", "_", 0, "root")

    # --- CASO 1: Stack pequeño (Necesita Padding) ---
    # Estado: Stack=[ROOT], Buffer=[Distribution, of]
    print("\n>>> Test Case 1: Stack has only 1 element (ROOT). Padding needed.")
    
    state_1 = State([t_root], [t_dist, t_of], set())
    sample_1 = Sample(state_1, Transition(ArcEager.SHIFT)) # La transición no importa aquí
    
    # Llamamos a la función
    features_1 = sample_1.state_to_feats(nbuffer_feats=2, nstack_feats=2)
    
    print(f"State Stack: {[t.form for t in state_1.S]}")
    print(f"State Buffer: {[t.form for t in state_1.B]}")
    print(f"Generated Features: {features_1}")

    # Verificación visual
    expected_1 = ['<PAD>', 'ROOT', 'Distribution', 'of', '<PAD>', 'ROOT_UPOS', 'NOUN', 'ADP']
    print(f"Match Expected? {features_1 == expected_1}")


    # --- CASO 2: Stack lleno (Sin Padding) ---
    # Estado: Stack=[ROOT, Distribution], Buffer=[license, does]
    print("\n>>> Test Case 2: Stack has 2 elements. No padding needed.")
    
    state_2 = State([t_root, t_dist], [t_lic, t_does], set())
    sample_2 = Sample(state_2, Transition(ArcEager.LA)) 

    # Llamamos a la función
    features_2 = sample_2.state_to_feats(nbuffer_feats=2, nstack_feats=2)
    
    print(f"State Stack: {[t.form for t in state_2.S]}")
    print(f"State Buffer: {[t.form for t in state_2.B]}")
    print(f"Generated Features: {features_2}")

    # Verificación visual
    expected_2 = ['ROOT', 'Distribution', 'license', 'does', 'ROOT_UPOS', 'NOUN', 'NOUN', 'AUX']
    print(f"Match Expected? {features_2 == expected_2}")

'''

if __name__ == "__main__":
    
    tree_john = [
        Token(0, "ROOT", "ROOT", "ROOT", "_", "_", -1, "_"),
        Token(1, "John", "John", "PROPN", "_", "_", 2, "nsubj"),   # Head: ate (2)
        Token(2, "ate", "eat", "VERB", "_", "_", 0, "root"),      # Head: ROOT (0)
        Token(3, "a", "a", "DET", "_", "_", 5, "det"),            # Head: apple (5)
        Token(4, "green", "green", "ADJ", "_", "_", 5, "amod"),   # Head: apple (5)
        Token(5, "apple", "apple", "NOUN", "_", "_", 2, "obj")    # Head: ate (2)
    ]

    print("**************************************************")
    print("* Arc-eager function (Test: John ate a green apple) *")
    print("**************************************************\n")

    arc_eager = ArcEager()
    
    print("1. Initial state for: 'John ate a green apple'")
    state = arc_eager.create_initial_state(tree_john)
    print(state)

    print (f"\nIs the initial state a valid final state (buffer is empty)? {arc_eager.final_state(state)}\n")

    transition_shift = Transition(arc_eager.SHIFT)
    arc_eager.apply_transition(state, transition_shift)
    print("State after applying ONE SHIFT transition:")
    print(state, "\n")

    gold_arcs = arc_eager.gold_arcs(tree_john)
    print (f"Set of gold arcs: {gold_arcs}\n\n")

    print("**************************************************")
    print("* Creating instances of the class Transition/Sample *")
    print("**************************************************")
    
    print(f"Sample Transition: {Transition(ArcEager.LA, 'nsubj')}")
    sample_instance = Sample(state, transition_shift)
    print("Sample instance created.")


    print("\n" + "="*60)
    print("* ORACLE TEST: John ate a green apple           *")
    print("="*60)
    
    try:
        samples = arc_eager.oracle(tree_john)
        
        print(f"\nSUCCESS! The oracle generated {len(samples)} training samples.")
        print("\nGenerated sequence of transitions:")
        
        for i, s in enumerate(samples):
            stack_str = "[" + ", ".join([t.form for t in s.state.S]) + "]"
            buffer_str = "[" + ", ".join([t.form for t in s.state.B]) + "]"
            print(f"{i+1:02d}. Stack: {stack_str:<25} Buffer: {buffer_str:<30} -> Action: {s.transition}")
            
        print("\nOracle validation passed: Generated arcs match gold arcs exactly.")

    except AssertionError as e:
        print(f"\nFAILED: The arcs generated by your oracle do not match the gold arcs.")
        print(f"Error message: {e}")
        print("Hint: Check your is_correct() functions logic.")
    except Exception as e:
        print(f"\nCRASHED: Your code has an unexpected error.")
        print(f"Error: {type(e).__name__}: {e}") '''