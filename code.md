# 关于policy
policy使用update函数更新，调用顺序为：process_fn -> learn -> post_process_fn(in base.ignore)。
事实上，tianshou把trainier包装成了一个迭代器，实际运行是通过run函数的deque实现迭代的。因此可以用for ___ in trainer:训练。
· tianshou会用collector收集数据。collecor将buffer作为成员变量，通过policy.forward()收集batch，键值对可以通过继承自定义实现(当然也可以policy函数里自己重新调用一遍。)。与collect函数返回值无关。
· trainer会调用_next_训练。仅当收集到的数据足够训练后进行。一个epoch代表一次迭代。实际上，trainer把update封装了，它先调用policy_update_fn，再调用update，但其实_next_里面都是一些统计信息的处理，所以直接修改update就行了。
epoch_stat:{
    'test_reward': -615.7996379700406, 
    'test_reward_std': 0.0, 
    'best_reward': -541.509712166197, 
    'best_reward_std': 0.0, 
    'best_epoch': 0, 
    'loss': 40197.11877441406, 'loss/clip': 248.92523106001318, 'loss/vf': 79896.53491210938, 'loss/ent': 7.370357990264893, 
    'gradient_step': 16, 'env_step':160, 
    'rew': -427.287843524257, 
    'len': 10, 'n/ep': 2, 'n/st': 20}                                                                                                                                                                                                                               
info:{
    'duration': '548.40s', 'train_time/model': '1.41s', 
    'test_step': 30, 'test_episode': 3, 'test_time': '86.43s', 'test_speed': '0.35 step/s', 
    'best_reward': -541.509712166197, 'best_result': '-541.51 ± 0.00', 
    'train_step': 160, 'train_episode': 16, 'train_time/collector': '460.55s', 'train_speed': '0.35 step/s'}