from abc import ABC, abstractmethod

from algorithms.evaluation import base_evaluation_function, evaluation_function
from world.game_state import GameState


class MultiAgentSearchAgent(ABC):
    """Clase base para los agentes de búsqueda adversaria."""

    def __init__(self, depth: int | str = 2) -> None:
        self.depth = int(depth)
        if self.depth < 1:
            raise ValueError("La profundidad debe ser al menos 1 ply")
        self.nodes_evaluated = 0

    @abstractmethod
    def get_action(self, state: GameState) -> str | None:
        raise NotImplementedError


class MinimaxAgent(MultiAgentSearchAgent):
    """Agente Minimax para el defensor MAX frente al intruso MIN."""

    def minimax(self, state: GameState, agent_index, depth):
      self.nodes_evaluated +=1
      if state.is_win():
        return 1000
      elif state.is_lose():
        return -1000
      elif depth == 0:
        return evaluation_function(state)
      else:
          actions = state.get_legal_actions(agent_index)
          if agent_index == 0:
            best_value = -float("inf")
          elif agent_index == 1 :
            best_value = float("inf")
          
          for action in actions:
            successor = state.generate_successor(agent_index, action)
            next_agent = (agent_index + 1) % state.get_num_agents()
            value = self.minimax(successor, next_agent, depth - 1)
            if agent_index == 0 and value>best_value:
              best_value = value
            elif agent_index == 1 and value<best_value:
              best_value=value
      return best_value
          
        
      
    def get_action(self, state: GameState) -> str | None:
        """
        Retorna la acción del defensor con mayor valor Minimax.

        El defensor es MAX (agente 0), el intruso es MIN (agente 1) y cada
        acción consume un ply. Debe respetar el orden de las acciones legales,
        usar evaluation_function en terminales y cortes, y contar cada estado
        procesado una vez en self.nodes_evaluated, incluida la raíz.

        Tips:
        - Use state.get_legal_actions(agent_index) y
          state.generate_successor(agent_index, action) para expandir el árbol.
        - Compruebe state.is_win(), state.is_lose() y el corte de profundidad;
          evalúe esos estados con evaluation_function(state).
        - El siguiente agente es (agent_index + 1) % state.get_num_agents().
          depth=1 incluye una acción de MAX y depth=2 una de MAX y una de MIN.
        - Reinicie las métricas y cuente una vez cada estado procesado, incluida
          la raíz. Retorne la acción de MAX y conserve la primera en los empates.
        """
        self.nodes_evaluated = 1
        best_value = -float("inf")
        actions = state.get_legal_actions(0)
        if actions :
          best_action = actions[0]
          
          for action in actions:
            successor = state.generate_successor(0, action)
            value = self.minimax(successor, 1, self.depth-1)
            
            if value > best_value:
              best_value = value
              best_action = action
          return best_action
        else:
          return None


class AlphaBetaAgent(MultiAgentSearchAgent):
    """Agente Minimax que evita explorar ramas mediante poda alfa-beta."""

    def get_action(self, state: GameState) -> str | None:
        """
        Retorna la acción de Minimax aplicando poda alfa-beta.

        Debe usar la misma profundidad, orden de acciones y función de
        evaluación que Minimax.

        Tips:
        - Conserve la misma estructura y casos base de MinimaxAgent.
        - Inicie alpha en -infinito y beta en +infinito, y páselos en las
          llamadas recursivas.
        - En MAX actualice alpha y corte si valor >= beta; en MIN actualice beta
          y corte si valor <= alpha.
        """


        
        self.nodes_evaluated = 0

        def alpha_beta(current_state, depth, agent_index, alpha, beta):
            self.nodes_evaluated += 1

            if current_state.is_win() or current_state.is_lose() or depth <= 0:
                return evaluation_function(current_state)

            actions = current_state.get_legal_actions(agent_index)
            if not actions:
                return evaluation_function(current_state)

            next_agent = (agent_index + 1) % current_state.get_num_agents()

            # Defensor MAX
            if agent_index == 0:
                value = float("-inf")

                for action in actions:
                    successor = current_state.generate_successor(agent_index, action)
                    value = max(
                        value,
                        alpha_beta(successor, depth - 1, next_agent, alpha, beta)
                    )

                    if value >= beta:
                        return value

                    alpha = max(alpha, value)

                return value

            # Intruso MIN
            value = float("inf")

            for action in actions:
                successor = current_state.generate_successor(agent_index, action)
                value = min(
                    value,
                    alpha_beta(successor, depth - 1, next_agent, alpha, beta)
                )

                if value <= alpha:
                    return value

                beta = min(beta, value)

            return value

        # La raíz también cuenta
        self.nodes_evaluated += 1

        if state.is_win() or state.is_lose():
            return None

        actions = state.get_legal_actions(0)
        if not actions:
            return None

        best_action = actions[0]
        best_value = float("-inf")

        alpha = float("-inf")
        beta = float("inf")
        next_agent = 1 % state.get_num_agents()

        for action in actions:
            successor = state.generate_successor(0, action)

            value = alpha_beta(
                successor,
                self.depth - 1,
                next_agent,
                alpha,
                beta
            )

            if value > best_value:
                best_value = value
                best_action = action

            alpha = max(alpha, best_value)

        return best_action
