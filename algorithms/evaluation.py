import math

from world.game_state import GameState


def base_evaluation_function(state: GameState) -> float:
    """
    Retorna la evaluación base entregada para desarrollar el punto 4.

    Esta función no forma parte del código que debe modificar el estudiante y
    permite probar Minimax antes de desarrollar la heurística del punto 5.
    """
    if state.is_win():
        return 1000.0
    if state.is_lose():
        return -1000.0
    return float(state.get_score())


def evaluation_function(state: GameState) -> float:
    """
    Evalúa un estado desde la perspectiva del defensor MAX.

    Debe conservar las utilidades terminales de la evaluación base y diseñar
    una valoración no trivial para estados de corte. Minimax y alfa-beta usan
    esta misma función al comparar sus decisiones en el punto 5.

    Tips:
    - Los estados terminales ya se resuelven antes del bloque TODO; diseñe allí
      únicamente la valoración de estados no terminales.
    - Consulte state.defender_position, state.intruder_position,
      state.pending_terminals, state.get_score() y state.get_legal_actions(0).
    - state.layout.distance(start, goal) calcula y almacena en caché la distancia
      real por el mapa respetando los muros.
    - Maneje conjuntos vacíos y distancias infinitas, y mantenga todo estado no
      terminal estrictamente entre -1000 y +1000.
    """
    if state.is_win() or state.is_lose():
        return base_evaluation_function(state)

    layout = state.layout
    defender = state.defender_position
    intruder = state.intruder_position
    pending = state.pending_terminals

    total_terminals = len(layout.critical_nodes)
    activated = total_terminals - len(pending)

    max_distance = layout.height + layout.width

    value = 100.0 * activated

    if pending:
        goal_distance = min(
            layout.distance(defender, terminal) for terminal in pending
        )
        if goal_distance == float("inf"):
            goal_distance = max_distance
        value -= 25.0 * goal_distance / max_distance

    threat_distance = layout.distance(defender, intruder)
    if threat_distance == float("inf"):
        threat_distance = max_distance
    value += 50.0 * threat_distance / max_distance

    mobility = len(state.get_legal_actions(0))
    value += 2.0 * mobility

    return max(-999.0, min(999.0, value))
