TASK: Replace the use of LastEvent with Markers


Brice mentioned using markers instead of tracking the last events if we use immediate command lists.  it looks like we can do this in LZ with zeCommandListAppendSignalEvent. so I was thinking as an example:
for special NULL stream (0) and regular streams 1 and 2, say we:
launch kernel in 1
launch kernel in 2
launch kernel in 0
"launch kernel in 0" has an implicit sync, so streams 1 and 2 must complete whatever they are doing before the kernel in stream 0 can run. thus we create two events and add them into the immediate command lists for stream 1 and stream 2 with zeCommandListAppendSignalEvent and then add these events as dependencies to launching kernel 0.
this is different from tracking the last events as with that we would need to keep track of the events in the command lists for streams 1 and 2 and pass them around. here we can just add an event to wait on on the fly.

1. Comment out things in OpenCL just to get it to compile, don't even test opencl for now. 
2. Ignore event collector for Level Zero for now and all related warnigns. 
3. Comment out all event sanity checks for now.
4. Delete LastEvent_
5. Comment out everything related to fences for now. 
5. In Level Zero instead of collecting LastEvent for all target queues, just enqueue a signal that waits on nothing. Use createSharedEvent for creating events. 

## QUESTIONS FOR CLARIFICATION:

1. **Scope of the change**: Are we only replacing LastEvent usage in Level Zero backend, or do we need to maintain compatibility with OpenCL backend as well?
   **ANSWER**: Removing all LastEvent, for now we are ignoring OpenCL and not running any OpenCL tests. We are commenting things out in OpenCL to get it to compile. 

2. **Marker approach**: The current `enqueueMarkerImpl()` already uses `zeCommandListAppendSignalEvent` - is this the exact approach you want to use, or do you want a different implementation?
   **ANSWER**: Yes, use zeCommandListAppendSignalEvent directly, don't use enqueueMarkerImpl for now.  

3. **Event collector**: You mentioned ignoring the event collector for Level Zero - does this mean we should disable the `CHIPEventMonitorLevel0` entirely, or just ignore warnings from it?
   **ANSWER**: Disable it entirely and ignore all warnings related to it for now. This will break callbacks but that is ok.

4. **Fence handling**: When you say "comment out everything related to fences", do you mean the `FencedCmdList` class and its usage, or just specific fence-related operations?
   **ANSWER**: Everything related to regular command lists and zeFence should be commented out and ignored for now. 

5. **Testing strategy**: Should we focus on getting the code to compile first, then gradually re-enable functionality, or do you want a working implementation from the start?
   **ANSWER**: Working implementation from the start. 

## FOLLOW-UP QUESTIONS:

6. **Immediate command lists**: Should we keep using the existing immediate command list infrastructure (`getCmdListImm()`, `getCmdListImmCopy()`) or do we need to modify how these are created/managed?
   **ANSWER**: Remove all calls to getCmdListImmCopy and just call getCmdListImm and then commit. DO THIS FIRST. this will simplify things since otherwise we would need to deal with markers in both of these immediate command lists.

7. **Event creation**: When you say "use createSharedEvent for creating events", should we replace all `createEvent()` calls with `createEventShared()` calls, or only in specific contexts?
   **CLARIFICATION**: 
   - `createEvent()` creates a raw event pointer that the caller owns and must manually manage
   - `createEventShared()` creates a `std::shared_ptr<Event>` that uses reference counting for automatic cleanup
   - For the marker approach, we want `createEventShared()` because multiple queues may reference the same marker event and we don't want manual memory management complexity
   - Recommendation: Replace all `createEvent()` calls with `createEventShared()` calls in the marker implementation

8. **Queue synchronization**: For the marker approach, should we create a marker event for each queue that needs synchronization, or can we use a single marker event across multiple queues?
   **CLARIFICATION**:
   - **Single marker approach**: Create one marker event and signal it from all queues that need synchronization. Pros: Simpler event management, fewer events to track. Cons: Less granular control, harder to debug which specific queue caused issues.
   - **Multiple markers approach**: Create a separate marker event for each queue that needs synchronization. Each queue signals its own marker. Pros: Better debugging, more granular control, clearer dependencies. Cons: More events to manage, slightly more complex.
   - **Recommendation**: Use multiple markers approach as it's more explicit and easier to debug for a working implementation from the start.
   Use multiple marker events.

9. **Error handling**: Since we're disabling fences and event collector, what should happen when operations that previously relied on these fail? Should we return success or specific error codes?
   **ANSWER**: Just ignore failures.

10. **Memory management**: Without the event collector, how should we handle event cleanup and memory management? Should we implement a simpler cleanup mechanism?
    **ANSWER**: Just don't cleanup events for now. We need to implement the sync and then we will focus on cleanup

## REFINED PLAN:

### Phase 1: Simplify Command List Usage (DO THIS FIRST)
- Comment out current calls to getCmdListImmCopy and call getCmdListImm instead.
- Use only `getCmdListImm()` and commit immediately
- This simplifies marker handling since we only deal with one command list type

### Phase 2: Disable OpenCL and Event Sanity Checks
- Comment out OpenCL backend compilation temporarily
- Disable event sanity checks (`isDeletedSanityCheck()` calls)
- Comment out event collector warnings

### Phase 3: Remove LastEvent Dependencies
- Remove `LastEvent_` member variable from `Queue` class
- Remove `LastEventMtx` mutex
- Remove `updateLastEvent()` calls throughout the codebase
- Remove `getLastEvent()` and `getLastEventNoLock()` methods

### Phase 4: Disable Fences and Event Collector
- Comment out `FencedCmdList` class and all its usage
- Disable `CHIPEventMonitorLevel0` entirely
- Comment out all `zeFence` related operations
- Ensure proper cleanup without LastEvent tracking

### Phase 5: Implement Marker-Based Synchronization
- Modify `addDependenciesQueueSync()` to use `zeCommandListAppendSignalEvent` directly
- Create marker events using `createEventShared()` for each queue that needs synchronization
- Collect these markers as dependencies for the target event
- Replace LastEvent collection logic with marker creation

### Phase 6: Testing and Validation
- Compile and test basic functionality
- Ensure all tests pass
- Verify marker-based synchronization works correctly

## NOTES:
- Ignore all failures from disabled components (fences, event collector)
- Don't implement event cleanup for now - focus on sync functionality first
- Need clarification on event creation strategy and single vs multiple marker approach

