"""
Algorithm plugin registry.
如果用户传入了配置 algo_name = "ppo"，系统只需调用 algo = create_algorithm("ppo") 就能创建一个 PPO 实例去训练。
代码完全不需要 from ppo import PPO
"""

from __future__ import annotations

from typing import Callable, Dict, Type

from nanoverl.algos.base import RLAlgorithm

# 这是一个空字典，用来充当注册表。它的 Key 是算法名称（如 "ppo"），Value 是算法对应的类本身（不是实例）。
_ALGORITHM_REGISTRY: Dict[str, Type[RLAlgorithm]] = {}

# 是一个类装饰器。当你在其他文件里写具体的算法类时，把它放在类的头上
# @register_algorithm("ppo")  # 执行时，会自动调用注册表代码，把 PPO 类存进字典
# class PPO(RLAlgorithm):
#     pass
# 相当于 decorator(PPO)，会把 PPO 类作为参数传给 decorator 函数，最终把 PPO 类注册到 _ALGORITHM_REGISTRY 字典里，Key 是 "ppo"，Value 是 PPO 类本身变成 {"ppo": PPO}
def register_algorithm(name: str) -> Callable[[Type[RLAlgorithm]], Type[RLAlgorithm]]:
    def decorator(cls: Type[RLAlgorithm]) -> Type[RLAlgorithm]:
        if name in _ALGORITHM_REGISTRY and _ALGORITHM_REGISTRY[name] is not cls:
            raise ValueError("Algorithm already registered: %s" % name)
        cls.name = name
        _ALGORITHM_REGISTRY[name] = cls
        return cls

    return decorator


def get_algorithm_class(name: str) -> Type[RLAlgorithm]:
    try:
        # 根据字符串名字（比如 "ppo"），从字典里拿出对应的类（PPO 类）
        return _ALGORITHM_REGISTRY[name]
    except KeyError as exc:
        raise ValueError(
            "Unsupported algorithm: %s. Available algorithms: %s" % (name, sorted(_ALGORITHM_REGISTRY))
        ) from exc


def create_algorithm(name: str) -> RLAlgorithm:
    # 拿到类之后，加个括号 ()，实例化这个类并返回
    return get_algorithm_class(name)()


__all__ = ["create_algorithm", "get_algorithm_class", "register_algorithm"]
